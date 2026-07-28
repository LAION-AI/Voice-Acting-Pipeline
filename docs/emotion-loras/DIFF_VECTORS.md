# Diff-vector analysis — per-dimension max/min LoRA (effect = LoRA@100% minus base, same prompt)

dimension                      MAX-raise LoRA                      Δ   MIN LoRA (lowers)                   Δ   spread
Amusement                      Amusement                       +1.80   Pain                            -0.11     1.91
Emotional_Numbness             Longing                         +0.81   Embarrassment                   -0.93     1.74
Concentration                  Fear                            -0.04   Teasing                         -1.70     1.66
Fatigue_Exhaustion             Awe                             +0.83   Amusement                       -0.61     1.44
Impatience_and_Irritability    Impatience_and_Irritability     +1.54   Doubt                           +0.13     1.42
Embarrassment                  Sadness                         +1.08   Distress                        -0.22     1.30
Intoxication_Altered_States_of_Consciousness Amusement                       +1.19   Contentment                     -0.04     1.24
Relief                         Affection                       +0.53   Concentration                   -0.69     1.22
Hope_Enthusiasm_Optimism       Jealousy_&_Envy                 +0.80   Sadness                         -0.29     1.09
Contemplation                  Infatuation                     +0.32   Concentration                   -0.65     0.97
Confusion                      Sourness                        +0.78   Shame                           -0.17     0.95
Anger                          Impatience_and_Irritability     +0.90   Longing                         -0.04     0.94
Fear                           Sadness                         +0.66   Triumph                         -0.27     0.93
Contentment                    Disgust                         +0.06   Hope_Enthusiasm_Optimism        -0.83     0.89
Disappointment                 Sadness                         +0.85   Intoxication_Altered_States_of_Consciousness  +0.00     0.85
Teasing                        Teasing                         +0.77   Distress                        -0.05     0.82
Pride                          Hope_Enthusiasm_Optimism        +0.61   Pleasure_Ecstasy                -0.18     0.79
Interest                       Jealousy_&_Envy                 +0.36   Sadness                         -0.33     0.69
Elation                        Amusement                       +0.51   Infatuation                     -0.16     0.67
Longing                        Sadness                         +0.30   Confusion                       -0.35     0.66
Infatuation                    Contentment                     +0.53   Sadness                         -0.09     0.62
Helplessness                   Sadness                         +0.44   Amusement                       -0.18     0.62
Doubt                          Sadness                         +0.43   Helplessness                    -0.17     0.60
Pleasure_Ecstasy               Amusement                       +0.51   Pain                            -0.09     0.60
Distress                       Sadness                         +0.53   Malevolence_Malice              -0.04     0.57
Sadness                        Sadness                         +0.33   Amusement                       -0.18     0.51
Affection                      Amusement                       +0.21   Bitterness                      -0.30     0.51
Astonishment_Surprise          Jealousy_&_Envy                 +0.46   Anger                           -0.05     0.51
Thankfulness_Gratitude         Sadness                         +0.03   Helplessness                    -0.44     0.47
Triumph                        Hope_Enthusiasm_Optimism        +0.27   Pain                            -0.19     0.46
Jealousy_&_Envy                Contentment                     +0.22   Embarrassment                   -0.13     0.35
Malevolence_Malice             Fear                            +0.22   Distress                        -0.07     0.29
Sexual_Lust                    Contentment                     +0.11   Infatuation                     -0.09     0.19
Bitterness                     Helplessness                    +0.19   Sexual_Lust                     -0.01     0.19
Pain                           Sadness                         +0.14   Infatuation                     -0.05     0.19
Sourness                       Bitterness                      +0.17   Pleasure_Ecstasy                -0.01     0.17
Shame                          Sadness                         +0.10   Elation                         -0.01     0.12
Contempt                       Triumph                         +0.11   Doubt                           -0.00     0.11
Awe                            Distress                        +0.08   Affection                       +0.00     0.08
Disgust                        Relief                          +0.01   Doubt                           -0.00     0.01

## Notable collateral effects (each LoRA's strongest side-effect on another emotion)
  Affection                      lowers Concentration (-0.76) · raises Impatience_and_Irritability (+0.95)
  Amusement                      lowers Concentration (-1.65) · raises Impatience_and_Irritability (+1.21)
  Anger                          lowers Concentration (-0.84) · raises Impatience_and_Irritability (+1.28)
  Astonishment_Surprise          lowers Concentration (-1.17) · raises Impatience_and_Irritability (+0.90)
  Awe                            lowers Concentration (-0.67) · raises Impatience_and_Irritability (+0.91)
  Bitterness                     lowers Concentration (-1.24) · raises Impatience_and_Irritability (+1.51)
  Concentration                  lowers Relief (-0.69) · raises Impatience_and_Irritability (+1.29)
  Confusion                      lowers Contemplation (-0.65) · raises Impatience_and_Irritability (+0.34)
  Contemplation                  lowers Concentration (-0.82) · raises Impatience_and_Irritability (+0.80)
  Contempt                       lowers Contentment (-0.41) · raises Impatience_and_Irritability (+0.68)
  Contentment                    lowers Concentration (-0.86) · raises Impatience_and_Irritability (+0.76)
  Disappointment                 lowers Concentration (-0.57) · raises Impatience_and_Irritability (+0.72)
  Disgust                        lowers Concentration (-0.65) · raises Impatience_and_Irritability (+0.76)
  Distress                       lowers Concentration (-1.34) · raises Impatience_and_Irritability (+1.29)
  Doubt                          lowers Concentration (-0.44) · raises Impatience_and_Irritability (+0.13)
  Elation                        lowers Concentration (-1.31) · raises Impatience_and_Irritability (+1.39)
  Embarrassment                  lowers Concentration (-0.95) · raises Impatience_and_Irritability (+1.08)
  Emotional_Numbness             lowers Concentration (-0.91) · raises Impatience_and_Irritability (+0.79)
  Fatigue_Exhaustion             lowers Contemplation (-0.46) · raises Amusement (+0.15)
  Fear                           lowers Contentment (-0.26) · raises Amusement (+0.61)
  Helplessness                   lowers Concentration (-0.71) · raises Impatience_and_Irritability (+0.69)
  Hope_Enthusiasm_Optimism       lowers Contentment (-0.83) · raises Impatience_and_Irritability (+0.94)
  Impatience_and_Irritability    lowers Concentration (-1.39) · raises Anger (+0.90)
  Infatuation                    lowers Concentration (-0.55) · raises Impatience_and_Irritability (+1.10)
  Interest                       lowers Contentment (-0.54) · raises Impatience_and_Irritability (+0.87)
  Intoxication_Altered_States_of_Consciousness lowers Concentration (-1.01) · raises Impatience_and_Irritability (+0.93)
  Jealousy_&_Envy                lowers Concentration (-1.27) · raises Impatience_and_Irritability (+1.43)
  Longing                        lowers Contentment (-0.62) · raises Emotional_Numbness (+0.81)
  Malevolence_Malice             lowers Concentration (-0.78) · raises Impatience_and_Irritability (+1.09)
  Pain                           lowers Contentment (-0.59) · raises Impatience_and_Irritability (+0.66)
  Pleasure_Ecstasy               lowers Concentration (-0.81) · raises Impatience_and_Irritability (+1.04)
  Pride                          lowers Concentration (-1.14) · raises Impatience_and_Irritability (+1.30)
  Relief                         lowers Contemplation (-0.35) · raises Impatience_and_Irritability (+0.15)
  Sadness                        lowers Concentration (-0.73) · raises Embarrassment (+1.08)
  Sexual_Lust                    lowers Concentration (-0.92) · raises Impatience_and_Irritability (+0.99)
  Shame                          lowers Contemplation (-0.31) · raises Impatience_and_Irritability (+0.54)
  Sourness                       lowers Concentration (-1.03) · raises Impatience_and_Irritability (+1.49)
  Teasing                        lowers Concentration (-1.70) · raises Amusement (+1.71)
  Thankfulness_Gratitude         lowers Concentration (-1.22) · raises Impatience_and_Irritability (+1.19)
  Triumph                        lowers Concentration (-0.85) · raises Amusement (+0.64)
