## Main Points

### Value based methods

* 1-step Q-learning is slow, because we only update "one step back".
* n-step Q-learning is much more efficient at learning, because all "steps" are updated.

### Policy based methods

* Approximates gradient descent.
* Results in lower variance due to a baseline.

### Asynchronous (advantage) actor-critic

* Policy based.
* Always keeps an estimate of the value function _V_.
* RMSProp where statistics _g_ are shared across different threads, are
much more robust than the other methods. Maybe we should use this optimiser.

## Questions

* What does "(0, 1]" mean?
* What exactly does it mean to take _max(a')Q(s', a'; theta)_?
* Is __Sarsa__ a single actor learner? 
* What is __epsilon greedy exploration__?
* What is a __entropy regularisation term__?
* What does the optimiser __RMSProp__ do?

## Notes

* We might be able to use some of the sources in this article, as our own sources.
* They use ALE (Arcade Learning Environment) instead of OpenAI's gym.
* We need to look up the training and evaluation protocol used in this article.
* Watch the A3C learn to play TORCS Race Car game.
* Research what __LSTM__ neural networks are.
