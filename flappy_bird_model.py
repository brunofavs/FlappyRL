#!/usr/bin/env python3

# Imports for RL
import numpy as np
import gymnasium as gym

#------------------------------------------------------------------------------

# Imports for Flappy Bird
from flappy_bird import *
#------------------------------------------------------------------------------



class FlappyBird(gym.Env):

    def __init__(self):

        # Initialize positions - will be set randomly in reset()

        # Define what the agent can observe, Placeholder
        self.observation_space = gym.spaces.Discrete(8)

        # Define what actions are available (0: do nothing, 1: flap)
        self.action_space = gym.spaces.Discrete(2)

    def _get_obs(self):
        """Convert internal state to observation format.

           Essential observations:
             - Bird's vertical position (y-coordinate)
             - Bird's vertical velocity (to predict trajectory) -> Right now the bird doesn't accelerate downwards, so velocity is constant unless flapping
             - Next pipe's horizontal distance (how far away)
             - Next pipe's gap vertical position (top/bottom of opening)

           Additional useful observations:
             - Bird's distance from top/bottom of screen (helps avoid boundaries)
             - Next pipe's gap size (if it varies)
             - Second upcoming pipe info (for planning ahead)
             - Current score or progress (can help with learning)

        Returns:
            dict: Observations 
        """

        bird_y = self.bird.y


        return {}

    def _get_info(self):
        """Compute auxiliary information for debugging.

        Returns:
            dict: Info with distance between agent and target
        """
        return { }


    # https://chatgpt.com/share/694b61f1-0cf4-8011-9687-2abc22e5df2b
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Start a new episode.

        Args:
            seed: Random seed for reproducible episodes
            options: Additional configuration (unused in this example)

        Returns:
            tuple: (observation, info) for the initial state
        """
        # IMPORTANT: Must call this first to seed the random number generator
        super().reset(seed=seed)

        # Placeholder pseudocode, might not be necessary
        # if game is active
        #     print('Game over! Score: %i' % score)
        #     pygame.quit()
        pygame.init()

        self.display_surface = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
        pygame.display.set_caption('Pygame Flappy Bird')

        self.clock = pygame.time.Clock()
        self.score_font = pygame.font.SysFont(None, 32, bold=True)  # default font
        self.images = load_images()

        # the bird stays in the same x position, so bird.x is a constant
        # center bird on screen
        self.bird = Bird(50, int(WIN_HEIGHT/2 - Bird.HEIGHT/2), 2,
                    (self.images['bird-wingup'], self.images['bird-wingdown']))

        self.pipes = deque()

        self.frame_clock = 0  # this counter is only incremented if the game isn't paused
        self.score = 0
        self.done = self.paused = False

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        """Execute one timestep within the environment.

        Args:
            action: The action to take (0-3 for directions)

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """

        self.clock.tick(FPS)

        # Handle this 'manually'.  If we used pygame.time.set_timer(),
        # pipe addition would be messed up when paused.
        if not (self.paused or self.frame_clock % msec_to_frames(PipePair.ADD_INTERVAL)):
            pp = PipePair(self.images['pipe-end'], self.images['pipe-body'])
            self.pipes.append(pp)

        # 1 == flap 0 == do nothing
        if action == 1:
            self.bird.msec_to_climb = Bird.CLIMB_DURATION
        if action == 0: # just for readability
            pass  # do nothing

        # check for collisions
        self.pipe_collision = any(p.collides_with(self.bird) for p in self.pipes)
        if self.pipe_collision or 0 >= self.bird.y or self.bird.y >= WIN_HEIGHT - Bird.HEIGHT:
            self.done = True

        for x in (0, WIN_WIDTH / 2):
            self.display_surface.blit(self.images['background'], (x, 0))

        while self.pipes and not self.pipes[0].visible:
            self.pipes.popleft()

        for p in self.pipes:
            p.update()
            self.display_surface.blit(p.image, p.rect)

        self.bird.update()
        self.display_surface.blit(self.bird.image, self.bird.rect)

        # update and display score
        for p in self.pipes:
            if p.x + PipePair.WIDTH < self.bird.x and not p.score_counted:
                self.score += 1
                p.score_counted = True

        score_surface = self.score_font.render(str(self.score), True, (255, 255, 255))
        score_x = WIN_WIDTH/2 - score_surface.get_width()/2
        self.display_surface.blit(score_surface, (score_x, PipePair.PIECE_HEIGHT))

        pygame.display.flip()
        self.frame_clock += 1

        reward = self.score

        self.score = 0 # reset score after each step to give incremental reward

        terminated = self.done
        truncated = False

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info
