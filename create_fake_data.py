import numpy as np

def gen_fake_data():
    fake_data = np.random.rand(30, 3, 32, 32).astype(np.float32)
    np.save("fake_data.npy", fake_data)


def gen_fake_label():
    fake_label = np.random.randint(0, 10, [30, 1]).astype(np.int64)
    print(fake_label)
    np.save("fake_label.npy", fake_label)

if __name__ == "__main__":
    gen_fake_data()
    gen_fake_label()