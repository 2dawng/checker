import pygame
from constants import *

class Button:
    def __init__(self, x, y, width, height, text):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = BUTTON_COLOR
        self.is_hovered = False

    def draw(self, win):
        # Draw gradient background
        color = BUTTON_HOVER_COLOR if self.is_hovered else BUTTON_COLOR
        if self.is_hovered:
            gradient_surface = pygame.Surface((self.rect.width, self.rect.height))
            for i in range(self.rect.height):
                ratio = i / self.rect.height
                r = int(BUTTON_GRADIENT[0][0] * (1 - ratio) + BUTTON_GRADIENT[1][0] * ratio)
                g = int(BUTTON_GRADIENT[0][1] * (1 - ratio) + BUTTON_GRADIENT[1][1] * ratio)
                b = int(BUTTON_GRADIENT[0][2] * (1 - ratio) + BUTTON_GRADIENT[1][2] * ratio)
                pygame.draw.line(gradient_surface, (r, g, b), (0, i), (self.rect.width, i))
            win.blit(gradient_surface, (self.rect.x, self.rect.y))
        else:
            pygame.draw.rect(win, color, self.rect, border_radius=5)
        pygame.draw.rect(win, (100, 100, 100), self.rect, 2, border_radius=5)
        font = pygame.font.Font(None, 36)
        text = font.render(self.text, True, WHITE)
        text_rect = text.get_rect(center=self.rect.center)
        win.blit(text, text_rect)

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if self.is_hovered:
                return True
        return False