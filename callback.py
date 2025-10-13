from stable_baselines3.common.callbacks import BaseCallback
import csv

# Callback to write the reward and timestep to a .csv file while training with SB3
class LearningCurveCallback(BaseCallback):
    def __init__(self, verbose=0, log_file="learning_curve.csv"):
        super(LearningCurveCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.actor_losses = []
        self.critic_losses = []
        self.log_file = log_file

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        rewards = self.locals.get("rewards")
        if dones is not None and any(dones):
            if len(self.model.ep_info_buffer) > 0:
                latest_info = self.model.ep_info_buffer[-1]
                self.episode_rewards.append(latest_info.get("r", 0.0))

        actor_loss = self.model.logger.name_to_value.get("train/actor_loss")
        critic_loss = self.model.logger.name_to_value.get("train/critic_loss")
        if actor_loss is not None:
            self.actor_losses.append(actor_loss)
        if critic_loss is not None:
            self.critic_losses.append(critic_loss)
        return True

    def _on_training_end(self):
        # Save rewards to CSV file
        with open(self.log_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Episode', 'Reward'])
            for i, reward in enumerate(self.episode_rewards):
                writer.writerow([i, reward])
