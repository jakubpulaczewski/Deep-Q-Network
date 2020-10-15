# Deep-Q-Network

The project implements Deep Q-Network (DQN) with experience replay as our deep reinforcement learning model to train the agent to play the space invaders game that was imported from the Open-AI gym retro. It is important to note that the DQN is an existing deep reinforcement model which is trained with a variant of Q-learning algoriothm using the RMSprop optimizer to update its parameters, as it was choosen by a group of Deemind researchers in the paper for the Nature [1,2].

# Deep Q-learning with experience replay

The architecture of the network takes the last 4 video frames and feeds them into convolutional layers that are followed by fully connected layers to compute the Q values for each action. In space invaders game,there are only 8 Q values since the agent can only perform 8 actions.

The Atari network architecture  is shown below. It should be noted that the network architecture was mainly based on the DeepMind articles [1,2]. The only difference is that this neural network takes an input of a 105 x 80 x 4 image whereas the network mentioned in the articles first pre-process the image to an image of 110 x 84 x 1 and then crops the image to 84 x 84 x 1, and lastly stacks 4 frames in order to achieve the input sof 84 x 84 x 4.

![](Images/network.PNG)

The reason why we would want to pre-process the image is because the input image is in the form of 210 x 160 x 3, and it consists of 3 dimensions because of the RGB values. Therefore, we want to reduce the complexity by reducing it to grayscale image which results in an input of 210 x 160 x 1, and then downscale the image size to 105 x 80 x 1 which helps a lot with speeding up the process of learning.

It was previously mentioned that we use a stack of 4 frames that are fed into our convolutional network. For example, let's consider an example where we feed a single frame of 105 x 80 x 1 to the network that is trying to learn the 'SpaceInvaders-Atari2600' game. With just one frame, it is impossible for an agent to know the direction of other agents' attacks or how fast it is going. Thus, using a stack of four frames it can give an agent the sense of motion and speed that is necessary for the network to have a full understanding of the envrionment.
