# LunarLander-v2-Solve (MTS contest) BEST SCORE: 231

![preview](https://github.com/furfa/LunarLander-v2-Solve/blob/master/img/preview.gif)

> Награда за переход от верхней части экрана к посадочной площадке и нулевой скорости составляет около 100..140 баллов. Если посадочный аппарат отходит от посадочной площадки, он теряет награду. Эпизод заканчивается, если посадочный модуль падает или останавливается, получая дополнительные -100 или +100 очков. Каждый контакт с землей оценивается в +10 очков. Использование главного двигателя стоит -0,3 балла за каждый кадр. Выполнение условия посадки дает дополнительные 200 баллов. Возможна посадка вне посадочной площадки. Топливо бесконечно, поэтому агент может научиться летать, а затем научиться приземляться. Доступны четыре отдельных действия: ничего не делать, запустить двигатель с левой ориентацией, главный двигатель, двигатель с правой ориентацией.
## Метрика
> LunarLander-v2 определяет «решение» как получение среднего вознаграждения 200 за 100 последовательных испытаний.

## Использованные нами библиотеки:
* pytorch
* numpy
* gym
* pickle (Для сохранения агентов)

## Что мы делали
* За бейзлайн взяли модель и реализацию DQN отсюда [ТЫЦ](https://github.com/philtabor/Youtube-Code-Repository/tree/master/ReinforcementLearning/DeepQLearning) (По факту от нее уже ничего не осталось), и адаптировали к нашей задаче. Получили ответ более чем за 1000 итераций. Также посмотрели решение второго места лидерборда [ТЫЦ](https://github.com/plopd/deep-reinforcement-learning/blob/master/dqn/Deep_Q_Network.ipynb)
* Далее изменили алгоритм обновления весов фиксированной модели. Вместо простого перекидывания, стали взвешивать. Это дало лучший скор в 800 итераций
```python
if self.replace_target_cnt is not None and \
    self.learn_step_counter % self.replace_target_cnt == 0:

    self.Q_next.load_state_dict(self.Q_eval.state_dict())
```

```python
def update_weight(self, model_from, model_to, tau):
        for from_p, to_p in zip(model_from.parameters(), model_to.parameters()):
            to_p.data.copy_(tau*from_p.data + (1.0-tau)*to_p.data)
```
* Переписали память на numpy массивы - это дало серьезный прирост к скорости обучения. И инициализировали ее размером бача(Делая случайные действия в среде).
* Далее начался тюнинг, самым значимым параметром был EPSILON (он контролировал начальное изучение среды), определяющий вероятность совершения случайного действия, также мы минимизировали его разными методами, т.е. линейно, экспоненциально и т.д. После оптимизации задача решалась за +- 600 итераций.
* В нашей задаче максимальный reward равен 100 и даётся он при успешной посадке.Наш алгоритм пытается получить максимальный Ревард за итерацию и начинает летать, пытаясь получить ревард за успешную посадку ещё раз.Таким образом, чтобы избежать потерь reward’ов 
за использования основного двигателя, мы заканчиваем эпизод, если наш «Луноход» приземлился на землю. Это улучшило скор до 550 эпизодов и сильно сократило время обучения.
```python
def kostil(reward):
    return (
        reward == 100 or 
        reward == -100 or 
        reward == 10 or 
        reward == 200
        )
```
* Чтобы смотреть, как обучается модель мы написали класс для временного включения визуализации из gymа, 
```python
def try_block(env, scores, pbar, visualize):
    try:
        main_loop(env, scores, pbar, visualize)
    except KeyboardInterrupt:
        inp = input("""
            o - Остановка обучения,
            +v - Включить визуализацию
            -v - Выключить визуализацию
        """)
        if inp == 'o':
            print("Остановка обучения.")
        elif inp == '+v':
            try_block(env, scores, pbar, True)
        elif inp == '-v':
            try_block(env, scores, pbar, False)
        else:
            print("Продолжаем :)")
            try_block(env, scores, pbar, visualize)
```
Например в данной ситуации модель переобучилась, т.к. mean score растет медленно, а модель пытается адаптироваться ко всем observationам.
![alt](https://github.com/furfa/LunarLander-v2-Solve/blob/master/img/605.png)

Ситуация исправляется хорошей начальной инициализацией памяти, и уменьшением скорости уменьшения epsilona (Добавлением большего количества рандомных действий). Это все позволяет лучше исследовать среду. 

После оптимизации: 
![alt](https://github.com/furfa/LunarLander-v2-Solve/blob/master/img/411.jpg)

* Далее мы последовали советам https://drive.google.com/file/d/0BxXI_RttTZAhVUhpbDhiSUFFNjg/view из презентации от DEEPMIND.

Изменили MSELOSS на HUBER.

Подобрали гиперпараметры и получили скор 231.

Так выглядит финальная модель.

```python
class HuberNet(nn.Module):
    def __init__(self, ALPHA, INPUT_SHAPE, OUTPUT_SHAPE):
        super().__init__()

        self.ALPHA = ALPHA

        self.model = nn.Sequential(
            nn.Linear(INPUT_SHAPE, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, OUTPUT_SHAPE),
        )

        # self.optimizer = optim.RMSprop(self.parameters(), lr=self.ALPHA, momentum=0.0001) # Tune this
        self.optimizer = optim.Adam(self.parameters(), lr=self.ALPHA) # Tune this

        self.loss = nn.SmoothL1Loss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')     
        self.to(self.device)

    def forward(self, observation):
        observation = torch.Tensor(observation).to(self.device)
        actions = self.model(observation)
        return actions
```

## Результаты:

![alt](https://github.com/furfa/LunarLander-v2-Solve/blob/master/img/231.jpg)

## Наилучший результат - 231 итерация.

### ./Scripts/MAIN_BEST.py 