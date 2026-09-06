import pygame

pygame.init()
pygame.mixer.init()

pygame.mixer.music.load("MusMus-BGM-172.mp3")

while True:
    if not pygame.mixer.music.get_busy():
        pygame.mixer.music.play()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            raise SystemExit
