from worker.self_play import SelfPlay

self_play = SelfPlay()
experiences = self_play.generate_experiences(1000)
print(experiences)
