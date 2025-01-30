<!-- i am gonna make a cute lil game for the first years to understand this ddpm paper i think
for 1st task, i am thinking of making a game where user gets an image and behind the scene there are 15-20 "noise" patterns made of lines (diagonals and straight) that get transposed over the image at our ease procedurally.


now at any given level, any of the N noise patterns can be on the image and the noisy image as well as the original image will be provided to the user.
the user's task therefore is to choose which out of the mcq noise options are there in the image.

In task 2, the user will be shown a simulation. infront of the user, starting from 1 to N, each individual noise pattern will be transposed on the image infront of the user. the user needs to memorize the pattern and after the simulation, choose the patterns first forwards and then backwards.
then in the final boss level, i'll have actual gaussian normal distributions as the noise 😂 and the user will decipher that -->


### **Game Structure**
1. **Task 1: Noise Pattern Identification**
   - User sees a noisy image and the original image side by side.
   - User selects which noise patterns (from MCQ options) are applied to the noisy image.
   - Feedback is provided on correctness.

2. **Task 2: Noise Pattern Memorization**
   - User watches a simulation where noise patterns are sequentially applied to an image.
   - After the simulation, the user must recall and select the patterns in the correct order (forwards and backwards).
   - Feedback is provided on accuracy.

3. **Boss Level: Gaussian Noise Deciphering**
   - User is shown an image with Gaussian noise applied.
   - User must identify the noise characteristics (e.g., intensity, distribution) or match it to a reference.

---

#### 1. **Image and Noise Generator**
   - Create a `SquareImage` component that displays the original and noisy images.
   - Create a `NoisePattern` component that generates the noise patterns (grids with lines).
   - Use a `NoiseGenerator` utility to procedurally apply noise patterns to the image.

#### 2. **Task 1: Noise Identification**
   - Create a `Task1` component that:
     - Displays the original and noisy images.
     - Provides MCQ options for noise patterns.
     - Handles user input and provides feedback.

#### 3. **Task 2: Noise Memorization**
   - Create a `Task2` component that:
     - Plays a simulation of noise patterns being applied sequentially.
     - Provides an interface for the user to select patterns in order.
     - Validates the user’s input and provides feedback.

#### 4. **Boss Level: Gaussian Noise**
   - Create a `BossLevel` component that:
     - Applies Gaussian noise to an image.
     - Provides tools for the user to analyze and decipher the noise.

#### 5. **Game Manager**
   - Create a `GameManager` component that:
     - Tracks the user’s progress through tasks.
     - Handles transitions between tasks.
     - Displays scores and feedback.


### **Next Steps**
1. Implement the `NoiseGenerator` utility to create and apply noise patterns.
2. Add feedback mechanisms for user selections.
3. Build the simulation for Task 2.
4. Integrate Gaussian noise for the Boss Level.
5. Style the app to make it visually appealing.