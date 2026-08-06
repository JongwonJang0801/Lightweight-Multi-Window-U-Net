# Lightweight Multi-Window U-Net for Data-Efficient Frame-Level Speaker Segmentation

## Dataset

Using only one dataset would make it difficult to determine whether performance 257
gains result from the proposed inductive bias or from regularities peculiar to that dataset. 258
For example, average utterance duration, pause distribution, speaker balance, channel 259
characteristics, and phonetic coverage can influence frame-level accuracy. The study 260
therefore repeats the full model comparison on independently constructed LibriSpeech and 261
WSJ0 mixtures. This design does not guarantee universal fairness or domain robustness, 262
because both sources are read English speech and the synthetic mixing rule is shared. It 263
does, however, improve experimental equity among models and strengthens the claim that 264
trends are not restricted to one corpus. All four architectures are compared at the same 265
waveform-count levels within each dataset.

### if you want using same mix data
change **path_of_dataset** in code
run **./local/mix_libri.py or ./local/mix_wsj.py** and **./local/mk_mfcc_list.py**
