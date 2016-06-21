import numpy as np
from numpy import log, sqrt, exp

pdf = lambda x: exp(-0.5 * x * x)

def inverse_pdf(p):
    return sqrt(-2 * log(p))


C = 256
r = 3.6541528853610087963519472518
Vr = 0.0049286732339746519
x = np.zeros(C)
x[0] = r
for i in range(1, C - 1):
    x[i] = inverse_pdf(pdf(x[i - 1]) + Vr / x[i - 1])

x = x[::-1]

# ki
ki_final = {}
scales = (2 ** 52, 2 ** 23)
names = ('double', 'float')
for name, scale in zip(names, scales):
    ki = np.zeros(C, np.uint64)
    prob_ratio = x[:-1] / x[1:]
    ki[1:] = np.round(prob_ratio * scale).astype(np.uint64)
    ki[0] = np.uint64(np.round(((x.max() * pdf(x.max())) / Vr) * scale))
    digits = 10 if name == 'float' else 18
    out = ["{0:#0{1}X}".format((ki[i]), digits) for i in range(C)]
    ki_final[name] = out

# wi
wi_final = {}
scales = (2 ** 52, 2 ** 23)
names = ('double', 'float')
for name, scale in zip(names, scales):
    wi = x.copy()
    wi[0] = Vr / pdf(x.max())
    wi_final[name] = wi / scale

# fi
fi_final = {'double': pdf(x), 'float': pdf(x)}

constants = {'ziggurat_nor_r': 3.6541528853610087963519472518,
             'ziggurat_nor_inv_r': 0.27366123732975827203338247596,
             'ziggurat_exp_r': 7.6971174701310497140446280481}

type_map = {'uint64': 'uint64_t', 'uint32': 'uint32_t', 'double': 'double',
            'float': 'float'}
extra_text = {'uint64': 'ULL', 'uint32': 'UL', 'double': '', 'float': 'f'}


def write(a, name, dtype):
    ctype = type_map[dtype]
    out = 'static const ' + ctype + ' ' + name + '[] = { \n'
    format_str = '{0: .20e}' if dtype in ('double', 'float') else '{0}'
    formatted = [format_str.format(a[i]) + extra_text[dtype] for i in range(len(a))]
    lines = len(formatted) // 4
    for i in range(lines):
        temp = '    ' + ', '.join(formatted[4 * i:4 * i + 4])
        if i < (lines - 1):
            temp += ',\n'
        else:
            temp += '\n'
        out += temp
    out += '};'
    return out


with open('./ziggurat_constants.h', 'w') as f:
    f.write(write(ki_final['double'], 'ki_double', 'uint64'))
    f.write('\n\n')
    f.write(write(wi_final['double'], 'wi_double', 'double'))
    f.write('\n\n')
    f.write(write(fi_final['double'], 'fi_double', 'double'))

    f.write('\n\n\n\n')
    f.write(write(ki_final['float'], 'ki_float', 'uint32'))
    f.write('\n\n')
    f.write(write(wi_final['float'], 'wi_float', 'float'))
    f.write('\n\n')
    f.write(write(fi_final['float'], 'fi_float', 'float'))
