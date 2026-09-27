import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def create_ca(size: tuple, initial_type: str = 'center') -> np.ndarray:
    """
    Создаёт начальное состояние клеточного автомата.
    """
    r, c = size
    ca = np.zeros((r, c), dtype=int)
    
    if initial_type == 'center':
        ca[r // 2, c // 2] = 1
    elif initial_type == 'random':
        ca = np.random.randint(0, 2, size=(r, c))
    elif initial_type == 'line':
        ca[r // 2, :] = 1
        
    return ca

def update_parity(ca: np.ndarray, boundary: str = 'periodic') -> np.ndarray:
    """
    Обновляет состояние КА по правилу чётности за один шаг.
    Использует векторизированные операции NumPy для высокой скорости.
    """
    # Окрестность фон Неймана: сдвиги массива на 1 клетку в каждую сторону
    if boundary == 'periodic':
        # Периодические границы (тор) - лекция, раздел 2.2, рекомендует операцию %
        up = np.roll(ca, shift=1, axis=0)
        down = np.roll(ca, shift=-1, axis=0)
        left = np.roll(ca, shift=1, axis=1)
        right = np.roll(ca, shift=-1, axis=1)
        
    elif boundary == 'fixed':
        # Фиксированные границы (за пределами поля считаем, что там 0)
        up = np.pad(ca[1:, :], ((0, 1), (0, 0)), mode='constant', constant_values=0)
        down = np.pad(ca[:-1, :], ((1, 0), (0, 0)), mode='constant', constant_values=0)
        left = np.pad(ca[:, 1:], ((0, 0), (0, 1)), mode='constant', constant_values=0)
        right = np.pad(ca[:, :-1], ((0, 0), (1, 0)), mode='constant', constant_values=0)
    else:
        raise ValueError("Неподдерживаемый тип граничных условий. Выберите 'periodic' или 'fixed'.")

    # Суммируем состояния 4-х соседей
    neighbor_sum = up + down + left + right
    
    # Правило чётности: если сумма нечётная -> 1, чётная -> 0
    # Это эквивалентно neighbor_sum % 2
    new_ca = neighbor_sum % 2
    
    return new_ca

def simulate(ca: np.ndarray, steps: int, boundary: str = 'periodic') -> list:
    """
    Запускает симуляцию на заданное количество шагов и сохраняет историю.
    """
    history = [ca.copy()]
    current_ca = ca
    
    for _ in range(steps):
        current_ca = update_parity(current_ca, boundary=boundary)
        history.append(current_ca.copy())
        
    return history

def plot_history(history: list, title: str = "Эволюция правила чётности"):
    """
    Визуализирует несколько ключевых кадров эволюции (как рекомендуется в лекции).
    """
    n_plots = min(9, len(history))
    # Выбираем равномерно распределённые шаги, включая 0 и последний
    indices = np.linspace(0, len(history) - 1, n_plots, dtype=int)
    
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()
    
    for idx, ax in enumerate(axes):
        if idx < n_plots:
            step = indices[idx]
            ax.imshow(history[step], cmap='gray', interpolation='nearest')
            ax.set_title(f'Шаг {step}')
            ax.axis('off')
            
    # Скрываем лишние подграфики, если шагов меньше 9
    for idx in range(n_plots, 9):
        axes[idx].axis('off')
        
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def create_animation(history: list, filename: str = 'parity_ca.gif'):
    """
    Создаёт и сохраняет анимацию эволюции (упоминание в лекции: "научитесь делать анимацию").
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    
    def update(frame):
        ax.clear()
        ax.imshow(history[frame], cmap='plasma', interpolation='nearest')
        ax.set_title(f'Шаг {frame}')
        ax.axis('off')
        return ax,

    anim = FuncAnimation(fig, update, frames=len(history), interval=100, blit=False)
    
    # Для сохранения требуется установленный pillow или imagemagick
    try:
        anim.save(filename, writer='pillow', fps=10)
        print(f"✅ Анимация успешно сохранена в {filename}")
    except Exception as e:
        print(f"⚠️ Не удалось сохранить анимацию (возможно, не установлен pillow): {e}")
        plt.show() # Показываем анимацию в окне, если сохранение не сработало
        
    plt.close()

# ==========================================
# Пример запуска
# ==========================================
if __name__ == "__main__":
    SIZE = (64, 64)
    STEPS = 30
    
    print("Инициализация клеточного автомата...")
    initial_ca = create_ca(SIZE, initial_type='center')
    
    print("Запуск симуляции с периодическими границами...")
    # Периодические границы позволяют увидеть красивый фрактальный узор (треугольник Серпинского),
    # не искажённый краями поля, о чём предупреждает лекция.
    history_periodic = simulate(initial_ca, STEPS, boundary='periodic')
    
    print("Визуализация результатов...")
    # plot_history(history_periodic, title="Правило чётности (Периодические границы)")
    
    # Раскомментируйте строку ниже, чтобы создать GIF-анимацию
    create_animation(history_periodic, r'D:\уник\материалы\КМ\output\parity_periodic.gif')
    
    # Для сравнения можно запустить с фиксированными границами
    # history_fixed = simulate(initial_ca, STEPS, boundary='fixed')
    # plot_history(history_fixed, title="Правило чётности (Фиксированные границы)")