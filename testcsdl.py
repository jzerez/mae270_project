import csdl_alpha as csdl
import numpy as np

recorder = csdl.Recorder(inline=True)
recorder.start()
# x = csdl.Variable(shape=(1,))

# x.value = 25

# y = csdl.Variable(value=2)
# b = 1j
# print(b.dtype)
# z = x+y**b
# print(z.value)

# xs = np.arange(10)
# xc = csdl.Variable(name='x', value=xs)

# xs = np.random.random((3,3))
# ps = np.ones((3, 1))
# row = np.array(((0, 0, 0, 1)))
# x_mat = np.array([[xs, ps], [row]])
x_mat  = np.random.random((4,4))

mat = csdl.Variable(name='mat', shape = (4, 4), value=0)
x = csdl.Variable(name='x', value=25)
y = x**mat

def skew_sym(twist):
    mat = csdl.Variable(shape=(4,4), value=0)
    w1 = twist.value[0]
    w2 = twist.value[1]
    w3 = twist.value[2]
    mat = mat.set(csdl.slice[0, 1], -w3)
    mat = mat.set(csdl.slice[0, 2], w2)
    mat = mat.set(csdl.slice[0, 3], twist.value[3])

    mat = mat.set(csdl.slice[1, 0], w3)
    mat = mat.set(csdl.slice[1, 2], -w1)
    mat = mat.set(csdl.slice[1, 3], twist.value[4])

    mat = mat.set(csdl.slice[2, 0], -w2)
    mat = mat.set(csdl.slice[2, 1], w1)
    mat = mat.set(csdl.slice[2, 3], twist.value[5])

    mat = mat.set(csdl.slice[-1, -1], 1)

    return mat

v = csdl.Variable(value=np.array([[1,2,3,4,5,6]]).T)
w = skew_sym(v)
# mat.value[0:3, 0:3] = xs
# mat.value[0:3, -1] = ps
# mat.value[-1, :] = row
# mat.value = x_mat

# j3_ang = j3_ang.set(csdl.slice[2], csdl.sin(theta))

recorder.stop()
print(mat.value)
print(y.value)
print(y.value[0, -1])
print(v.value)
print(w.value)
recorder.visualize_graph()