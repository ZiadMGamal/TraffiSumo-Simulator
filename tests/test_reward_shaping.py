from environment.reward_shaping import RewardShaper


def test_local_reward():
    shaper = RewardShaper()
    reward = shaper.compute_local_reward("A", pressure=5.0, total_wait=10.0, throughput_delta=2)
    assert reward < 0


def test_cooperative_rewards():
    shaper = RewardShaper()
    local = {"A": -10.0, "B": -5.0, "C": -8.0}
    coop = shaper.compute_cooperative_rewards(local)
    assert set(coop.keys()) == set(local.keys())
    for aid in local:
        assert coop[aid] <= 0


def test_fairness_penalty():
    shaper = RewardShaper(fairness_weight=0.1)
    rewards = {"A": -5.0, "B": -5.0}
    queues = {"A": 10.0, "B": 2.0}
    adjusted = shaper.apply_fairness_penalty(rewards, queues)
    assert adjusted["B"] < adjusted["A"]
