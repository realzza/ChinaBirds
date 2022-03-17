# ChinaBirds
## Todo
- [x] prepare a balanced dataloader
- [x] prepare a utt2wav and utt2label. [utt] [wav] [start]
- [x] write in multithread style
- [ ] run the VAD model
- [x] prepare a customized dataset
- [ ] make sure label2int's order is the same as initializing the sampler.
- [ ] the object that the sampler is working on is merely a feature, using a window to sample. There must be an encoding&decoding mechanism to sample from the same file.
- [ ] remove unnecessary part in `train.py` and config files.