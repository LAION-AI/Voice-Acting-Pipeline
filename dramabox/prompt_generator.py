"""Multi-GPU LLM batch prompt generation using torch.multiprocessing.

Each GPU worker loads the LLM once, then pulls batches of prompt requests
from a shared queue and returns generated text via a result queue.
"""
import os
import random
import time
from multiprocessing import Process, Queue, Event
from pathlib import Path

from .csv_io import CSV_FIELDNAMES, save_chunk
from .prompts import SYSTEM_INSTRUCTION, build_full_prompt
from .sampling import sample_voicenet, sample_archetype
from .taxonomy import (
    parse_voicenet_html, load_vocal_bursts, load_archetypes,
    format_vocal_bursts_block, load_emonet,
)
from .wordlists import get_word_list


def gpu_worker(gpu_id: int, work_queue: Queue, result_queue: Queue,
               ready_event: Event, model_name: str, dtype_str: str,
               batch_size: int, max_tokens: int, temperature: float,
               top_p: float):
    """Worker process: loads LLM on one GPU and processes batches."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"[GPU {gpu_id}] Loading {model_name}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch_dtype, device_map="cuda",
    )
    model.eval()
    device = next(model.parameters()).device
    print(f"[GPU {gpu_id}] Model loaded on {device}", flush=True)

    ready_event.set()

    processed = 0
    total_time = 0.0

    while True:
        batch_items = []
        try:
            item = work_queue.get(timeout=10)
            if item is None:
                break
            batch_items.append(item)
        except Exception:
            if work_queue.empty():
                break
            continue

        while len(batch_items) < batch_size:
            try:
                item = work_queue.get_nowait()
                if item is None:
                    batch_items.append(None)
                    break
                batch_items.append(item)
            except Exception:
                break

        real_items = [it for it in batch_items if it is not None]
        has_poison = len(real_items) < len(batch_items)

        if not real_items:
            if has_poison:
                break
            continue

        all_texts = []
        for idx, system_prompt, user_prompt in real_items:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            all_texts.append(text)

        tokenizer.padding_side = "left"
        inputs = tokenizer(
            all_texts, return_tensors="pt", padding=True,
            truncation=True, max_length=8192,
        ).to(device)

        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
            )
        elapsed = time.time() - t0
        total_time += elapsed

        input_len = inputs["input_ids"].shape[1]
        for i, (idx, _, _) in enumerate(real_items):
            new_tokens = outputs[i][input_len:]
            if tokenizer.pad_token_id is not None:
                mask = new_tokens != tokenizer.pad_token_id
                if mask.any():
                    last_real = mask.nonzero()[-1].item() + 1
                    new_tokens = new_tokens[:last_real]
            if len(new_tokens) > 0 and new_tokens[-1] == tokenizer.eos_token_id:
                new_tokens = new_tokens[:-1]

            text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            prompt_tokens = (inputs["attention_mask"][i] == 1).sum().item()
            result_queue.put((idx, text, prompt_tokens, len(new_tokens)))

        processed += len(real_items)
        if processed % (batch_size * 5) == 0:
            rate = processed / total_time if total_time > 0 else 0
            print(f"[GPU {gpu_id}] Processed {processed} samples, "
                  f"{rate:.1f} req/s", flush=True)

        torch.cuda.empty_cache()

        if has_poison:
            break

    if total_time > 0:
        print(f"[GPU {gpu_id}] Done. {processed} total, "
              f"{processed/total_time:.1f} req/s avg", flush=True)


def generate_all_prompts(config: dict) -> Path:
    """Run full prompt generation pipeline.

    1. Load taxonomies
    2. Sample all attributes
    3. Launch GPU workers
    4. Collect results and save CSV chunks

    Returns the output directory path.
    """
    pg = config.get("prompt_generation", {})
    sampling_cfg = config.get("sampling", {})
    output_cfg = config.get("output", {})

    total = pg.get("total_prompts", 1000)
    seed = pg.get("seed", 42)
    gpus = pg.get("gpus", [0])
    batch_size = pg.get("batch_size", 16)
    max_tokens = pg.get("max_tokens", 1024)
    temperature = pg.get("temperature", 1.0)
    top_p = pg.get("top_p", 0.95)
    model_name = pg.get("llm_model", "google/gemma-4-E4B-it")
    dtype_str = pg.get("llm_dtype", "bfloat16")

    chunk_size = output_cfg.get("chunk_size", 5000)
    outdir = Path(output_cfg.get("output_dir", "./output"))
    csv_prefix = output_cfg.get("csv_prefix", "dramabox")
    archetype_ratio = sampling_cfg.get("archetype_ratio", 0.20)

    mandatory_dim_codes = set(sampling_cfg.get("mandatory_dims", ["TEMP", "GEND", "AGEV"]))

    random.seed(seed)
    n_gpus = len(gpus)

    print("=" * 72, flush=True)
    print("  DRAMABOX PROMPT GENERATION", flush=True)
    print(f"  Total samples      : {total:,}", flush=True)
    print(f"  GPUs               : {gpus}", flush=True)
    print(f"  Batch size         : {batch_size}", flush=True)
    print(f"  Max tokens         : {max_tokens}", flush=True)
    print(f"  Chunk size         : {chunk_size:,}", flush=True)
    print(f"  Output dir         : {outdir}", flush=True)
    print(f"  Random seed        : {seed}", flush=True)
    print(f"  Archetype ratio    : {archetype_ratio:.0%}", flush=True)
    print(f"  Model              : {model_name} ({dtype_str})", flush=True)
    print("=" * 72, flush=True)

    # Load taxonomies
    print("\nLoading taxonomies...", flush=True)
    dp = config.get("data_paths", {})
    all_dims = parse_voicenet_html(Path(dp["voicenet_html"]))
    mandatory_dims = [d for d in all_dims if d["code"] in mandatory_dim_codes]
    optional_dims = [d for d in all_dims if d["code"] not in mandatory_dim_codes]
    temp_dim = next(d for d in all_dims if d["code"] == "TEMP")
    arou_dim = next(d for d in all_dims if d["code"] == "AROU")

    vb_taxonomy = load_vocal_bursts(Path(dp["vocal_bursts_json"]))
    vb_block = format_vocal_bursts_block(vb_taxonomy)
    archetypes = load_archetypes(Path(dp["archetypes_json"]))
    emonet = load_emonet(Path(dp["emonet_json"]))
    emotion_categories = list(emonet.keys())

    wordlists_dir = Path(dp.get("wordlists_dir", "data/wordlists"))

    print(f"  VoiceNet: {len(all_dims)} dims ({len(mandatory_dims)} mandatory, "
          f"{len(optional_dims)} optional)", flush=True)
    print(f"  Vocal bursts: {len(vb_taxonomy)} types", flush=True)
    print(f"  Archetypes: {len(archetypes)} genres", flush=True)
    print(f"  EmoNet: {len(emotion_categories)} categories", flush=True)

    # Pre-warm word lists
    for lang in config["_active_languages"]:
        wl = get_word_list(lang, wordlists_dir)
        print(f"  Word list [{lang}]: {len(wl)} words", flush=True)

    def _wordlist_fn(language):
        return get_word_list(language, wordlists_dir)

    # Phase 1: Sample all attributes
    print(f"\nPhase 1: Sampling {total:,} attribute sets...", flush=True)
    t0 = time.time()

    all_samples = []
    all_prompts = []
    path_a_count = 0
    path_b_count = 0

    for i in range(total):
        if random.random() < archetype_ratio:
            s = sample_archetype(archetypes, temp_dim, arou_dim,
                                 emotion_categories, config)
            path_b_count += 1
        else:
            s = sample_voicenet(mandatory_dims, optional_dims,
                                emotion_categories, config,
                                wordlist_fn=_wordlist_fn)
            path_a_count += 1
        all_samples.append(s)
        all_prompts.append(build_full_prompt(s, vb_block))

    print(f"  Done in {time.time()-t0:.1f}s. Path A: {path_a_count:,}, "
          f"Path B: {path_b_count:,}", flush=True)

    # Phase 2: Launch GPU workers
    print(f"\nPhase 2: Launching {n_gpus} GPU workers...", flush=True)

    work_queue = Queue(maxsize=total + n_gpus)
    result_queue = Queue(maxsize=total)
    ready_events = []

    workers = []
    for gpu_id in gpus:
        ready = Event()
        ready_events.append(ready)
        p = Process(
            target=gpu_worker,
            args=(gpu_id, work_queue, result_queue, ready,
                  model_name, dtype_str, batch_size, max_tokens,
                  temperature, top_p),
            daemon=True,
        )
        p.start()
        workers.append(p)

    print("Waiting for models to load...", flush=True)
    for i, ready in enumerate(ready_events):
        ready.wait()
        print(f"  GPU {gpus[i]} ready", flush=True)
    print("All workers ready!", flush=True)

    print(f"Enqueueing {total:,} prompts...", flush=True)
    for i in range(total):
        work_queue.put((i, SYSTEM_INSTRUCTION, all_prompts[i]))
    for _ in range(n_gpus):
        work_queue.put(None)

    # Phase 3: Collect results and save chunks
    print(f"\nPhase 3: Collecting results and saving chunks...", flush=True)
    t_start = time.time()

    results = {}
    collected = 0
    chunk_idx = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    errors = 0
    last_report = time.time()

    while collected < total:
        try:
            idx, text, prompt_tokens, completion_tokens = result_queue.get(timeout=300)
            results[idx] = (text, prompt_tokens, completion_tokens)
            collected += 1
            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens

            if text.startswith("ERROR"):
                errors += 1

            now = time.time()
            if now - last_report > 10:
                elapsed = now - t_start
                rate = collected / elapsed
                eta = (total - collected) / rate if rate > 0 else 0
                print(f"  Progress: {collected:,}/{total:,} ({collected/total*100:.1f}%) | "
                      f"Rate: {rate:.1f} req/s | ETA: {eta/60:.1f}min | "
                      f"Errors: {errors} | "
                      f"Tokens: {total_prompt_tokens:,}in/{total_completion_tokens:,}out",
                      flush=True)
                last_report = now

            while True:
                chunk_start = chunk_idx * chunk_size
                if chunk_start >= total:
                    break
                chunk_end = min(chunk_start + chunk_size, total)
                if all(i in results for i in range(chunk_start, chunk_end)):
                    rows = []
                    for i in range(chunk_start, chunk_end):
                        s = all_samples[i]
                        text_i, _, _ = results[i]
                        rows.append({
                            "global_idx": i,
                            "sampling_path": s["sampling_path"],
                            "archetype_info": s["archetype_info"],
                            "language": s["language"],
                            "accent": s["accent"],
                            "emotions": s["emotions"],
                            "word_count_target": s["word_count_target"],
                            "must_include_words": ", ".join(s["must_include_words"]) if s["must_include_words"] else "",
                            "flow_style": s["flow_style"],
                            "flow_forced_by_voicenet": s["flow_forced_by_voicenet"],
                            "emotion_alignment": s["emotion_alignment"],
                            "direction_style": s["direction_style"],
                            "vocal_bursts_enabled": s["vocal_bursts_enabled"],
                            "attributes_raw": s["attributes_raw"],
                            "dramabox_prompt": text_i,
                        })
                    path = save_chunk(rows, chunk_idx, outdir, csv_prefix)
                    chunk_errors = sum(1 for r in rows if r["dramabox_prompt"].startswith("ERROR"))
                    print(f"  [Chunk {chunk_idx:03d}] Saved -> {path} "
                          f"({chunk_errors} errors)", flush=True)
                    for i in range(chunk_start, chunk_end):
                        del results[i]
                    chunk_idx += 1
                else:
                    break

        except Exception as e:
            print(f"  Error collecting result: {e}", flush=True)
            alive = sum(1 for w in workers if w.is_alive())
            if alive == 0:
                print("  All workers have exited!", flush=True)
                break

    for w in workers:
        w.join(timeout=30)

    # Save remaining results
    if results:
        remaining_indices = sorted(results.keys())
        rows = []
        for i in remaining_indices:
            s = all_samples[i]
            text_i, _, _ = results[i]
            rows.append({
                "global_idx": i,
                "sampling_path": s["sampling_path"],
                "archetype_info": s["archetype_info"],
                "language": s["language"],
                "accent": s["accent"],
                "emotions": s["emotions"],
                "word_count_target": s["word_count_target"],
                "must_include_words": ", ".join(s["must_include_words"]) if s["must_include_words"] else "",
                "flow_style": s["flow_style"],
                "flow_forced_by_voicenet": s["flow_forced_by_voicenet"],
                "emotion_alignment": s["emotion_alignment"],
                "direction_style": s["direction_style"],
                "vocal_bursts_enabled": s["vocal_bursts_enabled"],
                "attributes_raw": s["attributes_raw"],
                "dramabox_prompt": text_i,
            })
        path = save_chunk(rows, chunk_idx, outdir, csv_prefix)
        print(f"  [Chunk {chunk_idx:03d}] Saved remaining -> {path}", flush=True)

    elapsed_total = time.time() - t_start
    print("\n" + "=" * 72, flush=True)
    print("  RUN COMPLETE", flush=True)
    print(f"  Total generated     : {collected:,}", flush=True)
    print(f"  Total errors        : {errors}", flush=True)
    print(f"  Total time          : {elapsed_total:.1f}s ({elapsed_total/60:.1f}min)", flush=True)
    if elapsed_total > 0:
        print(f"  Avg rate            : {collected / elapsed_total:.1f} req/s", flush=True)
    print(f"  Chunks written      : {chunk_idx + 1} files in {outdir}/", flush=True)
    print(f"  Path A (voicenet)   : {path_a_count:,} ({path_a_count/total*100:.1f}%)", flush=True)
    print(f"  Path B (archetype)  : {path_b_count:,} ({path_b_count/total*100:.1f}%)", flush=True)
    print(f"  Tokens              : {total_prompt_tokens:,} in / "
          f"{total_completion_tokens:,} out", flush=True)
    print("=" * 72, flush=True)

    return outdir
