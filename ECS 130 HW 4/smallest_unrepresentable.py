import numpy as np

# TODO part i: Determine the smallest number unrepresentable by a `float`
# with a brute force search. Hint: use a `while` loop that counts up from 0 and
# use `np.float32` to convert an integer to a floating point value.
#
# This search unfortunately is rather slow when implemented in Python (taking
# about 15s on my computer). While debugging your solution, you could try using
# `np.float16` instead of `np.float32`, since the smallest unrepresentable
# number is much smaller for that number type.

i = 0
while True: #we need to create a condition so that while 
    if np.float32(i) != i:
        break # used np.float32 so that it converts int to floating point value
    i += i; 
print(f'Smallest unrepresentable integer (float): {i}')

# TODO part ii: check your analytical answer of the smallest unrepresentable
# number `n` by checking which numbers in the neighborhood [n - 8, n + 8] are
# representable by `float64`.
# It's a little tricky to do this test in Python because of the way numpy
# constantly tries to convert things to `double`. For example if you have a
# number `x` stored as a `np.float64` and you want to check if it exactly equals an
# integer `i`, you can't just do `x == i` (this will first convert `i` to `double`)
# but instead need to do `int(x) == i`.

n = 0
for i in range(n -8, n + 8+1): #we need to modify this so that it adjusts to the correct rang
    x = np.float64(i)
    rep == int(x) == i #made sure that int(x) == i 
