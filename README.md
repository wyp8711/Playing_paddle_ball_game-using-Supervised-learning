# Playing_paddle_ball_game-using-Supervised-learning
Playing paddle ball game using Supervised Learning

# Date: 2019.08.16

# Reference source
https://blog.csdn.net/chencaw/article/details/78308685

# Training and Testing video

# Pre-requisites
* Python (tested on 3.5)
* Keras
* Tensorflow
* pygame

# How to training and testing
**1.Run "game_train.py"**

First. The program create random data for first run.

It's gonna train about 3 times, and restart game when score=500(Good for test).

It will close when restart game 10 times.

If you wanna train within the shortest possible time, you can comment ["self.clock.tick(60)"line 231]&["pygame.time.wait(600)"line 223].

**2.Run "game_auto(predict).py"**

If you wanna test within the shortest possible time, you can comment ["self.clock.tick(60)"line 106]&["pygame.time.wait(600)"line 98].


**There is a final model in "Final_model.zip".**
