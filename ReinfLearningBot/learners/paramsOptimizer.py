import optuna
from rlgym_tools.sb3_utils import SB3MultipleInstanceEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecCheckNan, VecMonitor, VecNormalize
from torch.nn import Tanh
from ReinfLearningBot.learners.gameplay_learner import get_match

if __name__ == '__main__':
    # CONFIG
    num_instances = 4
    agents_per_match = 2

    max_target_steps = 2_500_000
    max_steps = max_target_steps // (num_instances * agents_per_match)
    max_batch_size = max_target_steps // 10

    min_target_steps = 500_000
    min_steps = min_target_steps // (num_instances * agents_per_match)
    min_batch_size = min_target_steps // 10

    n_steps_step = 250_000 // (num_instances * agents_per_match)
    batch_size_step = 25_000

    # ENV
    env = SB3MultipleInstanceEnv(get_match, 4)  # Optional: add custom waiting time to load more instances
    env = VecCheckNan(env)
    env = VecMonitor(env)  # Useful for Tensorboard logging
    env = VecNormalize(env, norm_obs=True, gamma=0.999)


    def objective(trial):
        # Sunt trimise intervale de cautare pentru fiecare parametru
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 5e-4, step=2e-5)
        steps = trial.suggest_int("steps", min_steps, max_steps, step=n_steps_step)
        batch_size = trial.suggest_int("batch", min_batch_size, max_batch_size, step=batch_size_step)

        n_epochs = trial.suggest_int('n_epochs', 1, 31, step=5)
        vf_coef = trial.suggest_float('vf_coef', 0.5, 1., step=0.1)
        ent_coef = trial.suggest_float('ent_coef', 0., 0.2, step=0.05)
        clip_range = trial.suggest_float('clip_range', 0.1, 0.4, step=0.05)
        gae_lambda = trial.suggest_float('gae_lambda', 0.2, 1.0, step=0.2)

        layer_size = trial.suggest_int("neurons", 64, 256, step=64)
        layers_num = trial.suggest_int("layers", 1, 3)

        # Sunt optimizate numarul de layere si de noduri pentru retelele neurale
        policy_kwargs = dict(
            activation_fn=Tanh,
            net_arch=dict(pi=[layer_size] * layers_num, vf=[layer_size] * layers_num)
        )

        # Sunt optimizati hiperparametrii pentru PPO
        model = PPO(
            "MlpPolicy",
            env,
            n_epochs=n_epochs,
            policy_kwargs=policy_kwargs,
            learning_rate=learning_rate,
            ent_coef=ent_coef,  # From PPO Atari
            vf_coef=vf_coef,  # From PPO Atari
            gamma=0.999,
            verbose=3,
            batch_size=batch_size,
            n_steps=steps,
            clip_range=clip_range,
            gae_lambda=gae_lambda,
            tensorboard_log="./rl_tensorboard_log",
            device="auto"
        )

        training_steps_num = 30_000_000  # Numarul maxim de pasi al unui experiment
        training_epochs = 3
        mean_reward, std_reward = 0.0, 0.0

        # Bucla de antrenare
        # Antrenarea este impartita in 3 etape, astfel este posibila oprirea prematura
        # a experimentului dupa 10 sau dupa 20 de milioane de epoci
        for epoch in range(training_epochs):
            model.learn(training_steps_num // 3, tb_log_name="1s_config_1", reset_num_timesteps=False)

            # Este evaluata configuratia dupa fiecare 1/3 din durata experimentului
            mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=5000)

            # Oprire prematura daca progresul este prea mic pentru epoca curenta
            trial.report(mean_reward, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        print(f'Mean reward: {mean_reward}, STD: {std_reward}')
        return mean_reward


    study = optuna.create_study(direction="maximize")  # Se cauta maximizarea obiectivului
    study.optimize(objective, n_trials=100)  # Se ruleaza 100 de iteratii de optimizare

    # Se salveaza statisticile de optimizare
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    print("Study statistics: ")
    print("Number of finished trials: ", len(study.trials))
    print("Number of pruned trials: ", len(pruned_trials))
    print("Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print(" Value: ", trial.value)
    print(" Params: ", )
    for key, value in trial.params.items():
        print(f"{key}: {value}")
