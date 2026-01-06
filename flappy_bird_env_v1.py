#!/usr/bin/env python3

# Imports for RL
import numpy as np
import gymnasium as gym

#------------------------------------------------------------------------------

# Imports for Flappy Bird
from flappy_bird import *
#------------------------------------------------------------------------------



class FlappyBird_v1(gym.Env):

    def __init__(self):

        # Initialize positions - will be set randomly in reset()

        # Define what the agent can observe, Placeholder
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)

        # Define what actions are available (0: do nothing, 1: flap)
        # TODO: Change to 0 hold for a few frames, 1: flap 
        self.action_space = gym.spaces.Discrete(2)
        
        # # Initialize pygame and window once
        pygame.init()
        self.display_surface = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
        pygame.display.set_caption('Pygame Flappy Bird')
        self.clock = pygame.time.Clock()
        self.score_font = pygame.font.SysFont(None, 32, bold=True)
        self.images = load_images()

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

        # Bird vertical position
        bird_y = self.bird.y

        # Find the next pipe (first one that hasn't been passed)
        next_pipe = None
        for p in self.pipes:
            # There may be many pipes ahead, but it will find the first one and
            # break because its a deque and pipes are added from left to right
            if p.x + PipePair.WIDTH > self.bird.x:
                 next_pipe = p
                 break

        if next_pipe:
            pipe_distance = max(next_pipe.x - self.bird.x,0)             # Horizontal distance to next pipe

            gap_top = next_pipe.top_height_px                     # Y coordinate where gap starts
            gap_bottom = WIN_HEIGHT - next_pipe.bottom_height_px  # Y coordinate where gap ends
        else:
            # No pipes ahead, set to -1. This can happen at the very start.
            pipe_distance = -1
            gap_top       = -1
            gap_bottom    = -1

        # self.bird.y is distance from top of screen. Y increases downwards.
        # WIN_HEIGHT - (self.bird.y + Bird.HEIGHT) is distance from bottom of screen
        min_boundary_distance = min(self.bird.y, WIN_HEIGHT - (self.bird.y + Bird.HEIGHT))
        
        self.last_observation = np.array([bird_y,pipe_distance,gap_bottom,gap_top,min_boundary_distance], dtype=np.float32)

        return self.last_observation

    def _get_info(self):
        """Compute auxiliary information for debugging.

        Returns:
            dict: Info with labeled observation values
        """
        if not hasattr(self, 'last_observation') or self.last_observation is None:
            return {}
        
        return {
            'bird_y': float(self.last_observation[0]),
            'pipe_distance': float(self.last_observation[1]),
            'gap_bottom': float(self.last_observation[2]),
            'gap_top': float(self.last_observation[3]),
            'min_boundary_distance': float(self.last_observation[4]),
            'spam_penalty': float(getattr(self, 'spam_penalty', 0.0)),
            'distance_reward': float(getattr(self, 'distance_reward', 0.0)),
            'cumulative_reward': float(getattr(self, 'cumulative_reward', 0.0)),
            'last_step_reward': float(getattr(self, 'last_step_reward', 0.0)),
        }


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

        # Reset game state (no window recreation)
        # Placeholder pseudocode, might not be necessary
        # if game is active
        #     print('Game over! Score: %i' % score)
        #     pygame.quit()

                    # Moved to __init__    
                    # pygame.init()

                    # self.display_surface = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
                    # pygame.display.set_caption('Pygame Flappy Bird')

                    # self.clock = pygame.time.Clock()
                    # self.score_font = pygame.font.SysFont(None, 32, bold=True)  # default font
                    # self.images = load_images()

        # the bird stays in the same x position, so bird.x is a constant
        # center bird on screen
        self.bird = Bird(50, int(WIN_HEIGHT/2 - Bird.HEIGHT/2), 2,
                    (self.images['bird-wingup'], self.images['bird-wingdown']))

        self.pipes = deque()

        self.frame_clock = 0  # this counter is only incremented if the game isn't paused
        self.score = 0
        self.done = self.paused = False
        self.last_action = 0  # Track last action to prevent spam
        self.consecutive_flaps = 0  # Track consecutive flaps for spam penalty
        self.last_pipe_x = None  # Track last cleared pipe position

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        """Execute one timestep within the environment.
            action: The action to take (0-3 for directions)

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """

        self.clock.tick(FPS)
        cumulative_reward = 0.0

        # Handle this 'manually'.  If we used pygame.time.set_timer(),
        # pipe addition would be messed up when paused.
        if not (self.paused or self.frame_clock % msec_to_frames(PipePair.ADD_INTERVAL)):
            pp = PipePair(self.images['pipe-end'], self.images['pipe-body'])
            self.pipes.append(pp)

        # 1 == flap, 0 == do nothing
        # Only allow flap if bird is not currently climbing
        if action == 1 and self.bird.msec_to_climb <= 0:
            self.bird.msec_to_climb = Bird.CLIMB_DURATION
            self.consecutive_flaps += 1
            
            # Base flap penalty + escalating penalty for consecutive flaps
            base_penalty = 2
            self.spam_penalty = (self.consecutive_flaps ** 2) * 0.1  # quadratic penalty
            # print(f"Flap penalty: base {base_penalty} + spam {spam_penalty} (consecutive: {self.consecutive_flaps})")
            cumulative_reward -= (base_penalty + self.spam_penalty)
            
            self.last_action = 1
        else:
            self.last_action = 0

        if action == 0: # The consecutive counter has to reset when the agent chooses not to flap, not when the bird is climbing and can't flap.
            self.consecutive_flaps = 0  # Reset counter when not flapping

        # check for collisions
        self.pipe_collision = any(p.collides_with(self.bird) for p in self.pipes)
        if self.pipe_collision or 0 >= self.bird.y or self.bird.y >= WIN_HEIGHT - Bird.HEIGHT:
            cumulative_reward += -500.0  # negative reward for dying
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
                
                #BUG: This will be used when I decide to have variable pipe spacing/speed, but its bugged atm. 
                # pipe_distance = p.x for some reason

                # # Calculate dynamic reward based on distance traveled
                # if self.last_pipe_x is not None:
                #     pipe_distance = abs(p.x - self.last_pipe_x)
                #     # Reward proportional to distance: normalize by expected distance
                #     # At 60 FPS, 3000ms interval, 0.18 px/ms speed = 540 px spacing
                #     expected_distance = 540  # approximate default spacing
                #     self.distance_reward = (pipe_distance / expected_distance) * 100
                #     print(p.x)
                #     print(self.last_pipe_x)
                #     print(pipe_distance)
                #     print(expected_distance)
                #     print(self.distance_reward)
                #     cumulative_reward += self.distance_reward
                #     input("waiting")
                # else:
                #     # First pipe, use default reward
                #     cumulative_reward += 100

                cumulative_reward += 300.0  
                
                self.last_pipe_x = p.x
                p.score_counted = True

        score_surface = self.score_font.render(str(self.score), True, (255, 255, 255))
        score_x = WIN_WIDTH/2 - score_surface.get_width()/2
        self.display_surface.blit(score_surface, (score_x, PipePair.PIECE_HEIGHT))

        pygame.display.flip()
        self.frame_clock += 1

        cumulative_reward += 0.01  # reduced survival reward (was 0.1)
        self.last_step_reward = cumulative_reward

        # self.score = 0 # reset score after each step to give incremental reward

        terminated = self.done
        truncated = False

        observation = self._get_obs()
        info = self._get_info()

        return observation, cumulative_reward, terminated, truncated, info
