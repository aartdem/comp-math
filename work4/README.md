# Метод конечных элементов

Все измерения проводились на следующем оборудовании:

* Процессор - 11th Gen Intel® Core™ i5-11500 @ 2.70GHz (6 ядер);
* Оперативная память - 16ГБ;
* Операционная система - Ubuntu 22.04.4 LTS;
* Версия gcc - 11.4.0;
* Версия OpenMP - 4.5.

## Рассматриваемое уравнение

Решалось уравнение $y'' - \lambda y = - 2 \lambda \sin (\sqrt \lambda x) $, которое 
удовлетворяет общему виду уравнения 8.4 из книги
[Методы вычислений](http://www.ict.nsc.ru/matmod/files/textbooks/KhakimzyanovCherny-2.pdf).
В нашем случае получается, что $p(x)=1, q(x)=\lambda , f(x)= 2 \lambda \sin (\sqrt \lambda x)$.
Также использовались граничные условия Дирихле и равномерная сетка.
При этом уравнение имеет решение, которое выражается явно: $y = \sin (\sqrt \lambda x)$.
## Сходимость 
Рассмотрим теорему 8.5 из пособия: $$\lvert\lvert y-y_h\rvert\rvert \leq (c c')^2 * h^2 \lvert\lvert f \rvert \rvert, \ норма - L_{2[0;l]} $$
 h - расстояние между соседними узлами сетки. В нашем случае $$c = \frac 1 c_1 \big( (Q \frac l 2 + P_1) \frac l {2c_1} + 1 \big) = \frac {\lambda l^2} 4 +1$$

$$c' = J_m \sqrt {P + Ql^2 / 4}=J_m \sqrt {1 + \lambda l^2 / 4}$$

$J_m$ - константа из 7.3.

Алгоритм запускался для разных значений $N$ и $l$:
```
N = 10 | l = 3.141592653589793 | err = 0.048633532946589786
N = 10 | l = 6.283185307179586 | err = 0.048633532946589786
N = 20 | l = 3.141592653589793 | err = 0.015253429138775055
N = 20 | l = 6.283185307179586 | err = 0.015253429138775055
N = 30 | l = 3.141592653589793 | err = 0.008025723067873986
N = 30 | l = 6.283185307179586 | err = 0.008025723067873986
```
Используя посчитанные значения $c$ и $c'$ для каждого конкретного случая, было установлено, что при $J_m=0.1$ теорема выполнятеся, из чего следует,
что $\lvert\lvert y-y_h\rvert\rvert = O(h^2)$.

Теоретическая оценка о втором порядке сходимости подтверждается.