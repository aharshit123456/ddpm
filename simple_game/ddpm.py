# import pygame
# import random

# # Initialize Pygame
# pygame.init()

# # Screen dimensions
# WIDTH, HEIGHT = 800, 600
# screen = pygame.display.set_mode((WIDTH, HEIGHT))
# pygame.display.set_caption("DDPM Noise Game")

# # Colors
# WHITE = (255, 255, 255)
# BLACK = (0, 0, 0)

# # Load image
# image = pygame.image.load("estelle-peplum-top-tops-509.webp")
# image = pygame.transform.scale(image, (300, 300))

# # Generate noise patterns
# def generate_noise_pattern():
#     pattern = pygame.Surface((300, 300), pygame.SRCALPHA)
#     for _ in range(50):  # Draw random lines
#         start = (random.randint(0, 300), random.randint(0, 300))
#         end = (random.randint(0, 300), random.randint(0, 300))
#         pygame.draw.line(pattern, BLACK, start, end, 2)
#     return pattern

# noise_patterns = [generate_noise_pattern() for _ in range(20)]

# # Task 1: Identify the Noise
# def task1():
#     noisy_image = image.copy()
#     current_noise = random.choice(noise_patterns)
#     noisy_image.blit(current_noise, (0, 0))

#     screen.fill(WHITE)
#     screen.blit(image, (50, 50))
#     screen.blit(noisy_image, (400, 50))
#     pygame.display.flip()

#     # Wait for user input
#     running = True
#     while running:
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 pygame.quit()
#                 return
#             if event.type == pygame.KEYDOWN:
#                 if event.key == pygame.K_1:  # Example: User selects pattern 1
#                     print("You selected Pattern 1")
#                     running = False

# # Task 2: Memorize the Sequence
# def task2():
#     sequence = random.sample(noise_patterns, 5)  # Random sequence of 5 patterns
#     for pattern in sequence:
#         screen.fill(WHITE)
#         screen.blit(image, (50, 50))
#         # noisy_image = image.copy()
#         # screen.blit(pattern, (0, 0))
#         screen.set_alpha(0)
#         screen.blit(pattern, (400, 50))
#         pygame.display.flip()
#         pygame.time.wait(1000)  # Show each pattern for 1 second

#     # Ask user to recall the sequence
#     print("Recall the sequence of patterns!")

# # Main loop
# task1()
# task2()
# pygame.quit()