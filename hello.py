def func(i, x):
    return -(2/3)*x**2 + (2/3)*i*x -(7+i)/3


# curve_num = 1
# x_val = 1
# for iteration in range(200):
#     result = func(curve_num, x_val)
#     print(f'curve number: {curve_num}; x = {x_val}; vertex: {5000-result}')
#     curve_num += 1
#     x_val += 0.5

for i in [1742, 1705, 1680, 1667]:
    print(5001-i)