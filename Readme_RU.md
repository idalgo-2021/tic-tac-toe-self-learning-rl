# Крестики-нолики. Самообучение с подкреплением.

Представлено несколько методов самообучения с подкреплением (**Reinforcement learning**) сети(многосвязный перцептрон) игре в крестики-нолики.

## Особенности

* Архитектура нейросети: 
    * TD-методы и MC: `9 → hidden(27) → 1`, `18 → hidden(27) → 1`
    * Q-learn: `18 → hidden(36) → 1`
* Реализация: Python + NumPy    
* Обучение: через self-play с ручная реализацией backpropagation 
* Методы RL:
    * Backward TD(0)
    * Batch TD(0) 
    * Online TD(0)
    * Monte Carlo
    * Q-learning
* Сохранение и загрузка обученной модели
* Игра человека против обученной нейросети


## Обучение сети

```bash
# Запуск самообучения(число циклов - по-умолчанию)
# python3 [filename] train
python3 tic_tac_toe_backwardTD.py train  # например


# Чтобы указать число циклов самообучения используйте ключ --episodes, например
# python3 [filename] train --episodes 30000
```

## Игра с человеком

```bash
# Запуск для пошаговой игры пользователя с программой.
# python3 [filename] play
python3 tic_tac_toe_backwardTD.py play  # например


# Для указания используемой модели - укажите имя файла модели с ключом --load, например
# python3 [filename] play --load model.npz
```
