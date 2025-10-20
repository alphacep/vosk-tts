This is a heavily modified codebase from Matcha, StableTTS and StyleTTS2

https://github.com/shivammehta25/Matcha-TTS

https://github.com/KdaiP/StableTTS

https://github.com/yl4579/StyleTTS2

### Some important changes in last design:

  * We use voicebox-style guidance for flow matching (basically condition with phone embeddings), not mel like in Matcha. This adds more variability to inputs.
  * We use kaldi-predicted durations instead of monotonic align. The latter doesn't properly model pauses and does many bad things for multispeaker. The label files with durations look like below and created with nnet3-align software. Note that we use non-standard frame shift of 0.011 second (correspondin to hop size 256 at 22050 sample rate.
  * We use advanced frontend with punctuation embedding and word-position dependent phones
  * We train prior and duration embeddings separately (flow matching has high gradients and badly affects prior). We yet to implement the code to cleanly separate training steps. https://t.me/speechtech/2153
  * We add dither noise to mel to deal with zero energy regions common in dirty data.
 
### Label file example

```
^ 0 25
i0 25 11
z 36 8
n 44 5
u0 49 6
t 55 9
rj 64 5
i1 69 8
  77 5
d 78 3
o0 81 4
nj 85 6
e0 91 6
s 97 10
lj 107 6
i1 113 8
sj 121 11
$ 208 1
```
