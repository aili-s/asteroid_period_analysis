import pandas as pd
import matplotlib.pyplot as plt

# Зчитуємо CSV файли (увага: десяткові роздільники - коми)
df1 = pd.read_csv(r"D:\Робочий стіл\унік\диплом\текст диплому\дані для гарних графіків\my.csv", sep=';')
df2 = pd.read_csv(r"D:\Робочий стіл\унік\диплом\текст диплому\дані для гарних графіків\Lighcurve_data.csv")

print("Перший датафрейм (objects):")
print(df1.head())
print("\nДругий датафрейм (asteroids):")
print(df2.head())

# Сповпці
df1 = df1[['id', 'per', 'mag', 'Frq']].copy()
df2 = df2[['Number', 'H', 'P']].copy()
df2 = df2[~((df2['P'] < 2) & (df2['H'] >= 12) & (df2['H'] <= 19))].copy() # виправлення помилки у вихідних даних. В цьому місці не повинно бути точок, через існування бар'єру

# Побудова графіка
plt.figure(figsize=(10, 6))
plt.yscale('log')

# Бар'єри
plt.axhline(y=2.2, color='green', linestyle='--', linewidth=2, alpha=0.9)
plt.axvline(x=20.5, color='green', linestyle='--', linewidth=2, alpha=0.9)
# Графік для другого файлу (всі астероїди)
plt.scatter(df2['H'], df2['P'], alpha=0.7, label='Дані із LCDB', color='blue', s = 4)

# Графік для першого файлу (досліджувані об'єкти)
plt.scatter(df1['mag'], df1['per'], alpha=0.7, label='Отримані значення', color='red', marker='x', s = 10)

plt.xlabel('Абсолютна зоряна величина (mag)')
plt.ylabel('Період (год)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
#plt.xlim(-0.05, 1.5)
#plt.ylim(-3, 25)
plt.show()
