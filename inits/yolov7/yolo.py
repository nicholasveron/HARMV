# type: ignore
from models.experimental import *
from models.common import *
import torch


class MT(nn.Module):

    def forward(self, x):
        z = []
        za = []

        i = 0

        attn_func = self.attn_m[i]
        attn_res = attn_func.forward(x[0][i])  # conv

        bs, _, ny, nx = attn_res.shape  # x(bs,2352,20,20) to x(bs,3,20,20,784)
        reshape_attn_res = attn_res.view(bs, self.na, self.attn, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

        m_func = self.m[i]
        x_i = m_func.forward(x[0][i])  # conv

        bs, _, ny, nx = x_i.shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
        reshaped_x_i = x_i.view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

        xv = torch.arange(nx).repeat(ny, 1)
        yv = torch.arange(ny).repeat(nx, 1).transpose(0, 1)
        grid = torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).to(x[0][0].device.type)

        y = reshaped_x_i.sigmoid()
        y[..., 0:2] = (y[..., 0:2] * 3. - 1.0 + grid) * 8  # xy
        y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * torch.tensor([[[[[12,  16]]],
                                                              [[[19,  36]]],
                                                              [[[40,  28]]]]]).to(x[0][0].device.type)  # wh

        z.append(y.view(bs, -1, self.no))
        za.append(reshape_attn_res.view(bs, -1, self.attn))

        i = 1

        attn_func = self.attn_m[i]
        attn_res = attn_func.forward(x[0][i])  # conv

        bs, _, ny, nx = attn_res.shape  # x(bs,2352,20,20) to x(bs,3,20,20,784)
        reshape_attn_res = attn_res.view(bs, self.na, self.attn, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

        m_func = self.m[i]
        x_i = m_func.forward(x[0][i])  # conv

        bs, _, ny, nx = x_i.shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
        reshaped_x_i = x_i.view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

        xv = torch.arange(nx).repeat(ny, 1)
        yv = torch.arange(ny).repeat(nx, 1).transpose(0, 1)
        grid = torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).to(x[0][0].device.type)

        y = reshaped_x_i.sigmoid()
        y[..., 0:2] = (y[..., 0:2] * 3. - 1.0 + grid) * 16  # xy
        y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * torch.tensor([[[[[36,  75]]],
                                                              [[[76,  55]]],
                                                              [[[72, 146]]]]]).to(x[0][0].device.type)  # wh

        z.append(y.view(bs, -1, self.no))
        za.append(reshape_attn_res.view(bs, -1, self.attn))

        i = 2

        attn_func = self.attn_m[i]
        attn_res = attn_func.forward(x[0][i])  # conv

        bs, _, ny, nx = attn_res.shape  # x(bs,2352,20,20) to x(bs,3,20,20,784)
        reshape_attn_res = attn_res.view(bs, self.na, self.attn, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

        m_func = self.m[i]
        x_i = m_func.forward(x[0][i])  # conv

        bs, _, ny, nx = x_i.shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
        reshaped_x_i = x_i.view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

        xv = torch.arange(nx).repeat(ny, 1)
        yv = torch.arange(ny).repeat(nx, 1).transpose(0, 1)
        grid = torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).to(x[0][0].device.type)

        y = reshaped_x_i.sigmoid()
        y[..., 0:2] = (y[..., 0:2] * 3. - 1.0 + grid) * 32  # xy
        y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * torch.tensor([[[[[142, 110]]],
                                                              [[[192, 243]]],
                                                              [[[459, 401]]]]]).to(x[0][0].device.type)  # wh

        z.append(y.view(bs, -1, self.no))
        za.append(reshape_attn_res.view(bs, -1, self.attn))

        test_o = torch.cat(z, 1)
        attn_o = torch.cat(za, 1)

        # reshaping bases to stack with test and attn
        bases_o = torch.cat([x[1], x[2]], dim=1)

        # flatten then add zeropad
        bases_o = bases_o.flatten()
        target = test_o.shape[1]
        pad_size = target - (len(bases_o) % target)
        bases_o = torch.cat([bases_o, torch.zeros(pad_size).to(x[0][0].device.type)])
        bases_o = bases_o.view(1, target, -1)

        output = torch.cat([test_o, attn_o, bases_o], 2)
        return output


class Model(nn.Module):
    def forward(self, x):
        return self.forward_once(x)

    def forward_once(self, x):
        x_0 = self.model[0](x)
        x_1 = self.model[1](x_0)
        x_2 = self.model[2](x_1)
        x_3 = self.model[3](x_2)
        x_4 = self.model[4](x_3)
        x_5 = self.model[5](x_3)
        x_6 = self.model[6](x_5)
        x_7 = self.model[7](x_6)
        x_8 = self.model[8](x_7)
        x_9 = self.model[9](x_8)
        x_10 = self.model[10]([x_9, x_7, x_5, x_4])
        x_11 = self.model[11](x_10)
        x_12 = self.model[12](x_11)
        x_13 = self.model[13](x_12)
        x_14 = self.model[14](x_11)
        x_15 = self.model[15](x_14)
        x_16 = self.model[16]([x_15, x_13])
        x_17 = self.model[17](x_16)
        x_18 = self.model[18](x_16)
        x_19 = self.model[19](x_18)
        x_20 = self.model[20](x_19)
        x_21 = self.model[21](x_20)
        x_22 = self.model[22](x_21)
        x_23 = self.model[23]([x_22, x_20, x_18, x_17])
        x_24 = self.model[24](x_23)
        x_25 = self.model[25](x_24)
        x_26 = self.model[26](x_25)
        x_27 = self.model[27](x_24)
        x_28 = self.model[28](x_27)
        x_29 = self.model[29]([x_28, x_26])
        x_30 = self.model[30](x_29)
        x_31 = self.model[31](x_29)
        x_32 = self.model[32](x_31)
        x_33 = self.model[33](x_32)
        x_34 = self.model[34](x_33)
        x_35 = self.model[35](x_34)
        x_36 = self.model[36]([x_35, x_33, x_31, x_30])
        x_37 = self.model[37](x_36)
        x_38 = self.model[38](x_37)
        x_39 = self.model[39](x_38)
        x_40 = self.model[40](x_37)
        x_41 = self.model[41](x_40)
        x_42 = self.model[42]([x_41, x_39])
        x_43 = self.model[43](x_42)
        x_44 = self.model[44](x_42)
        x_45 = self.model[45](x_44)
        x_46 = self.model[46](x_45)
        x_47 = self.model[47](x_46)
        x_48 = self.model[48](x_47)
        x_49 = self.model[49]([x_48, x_46, x_44, x_43])
        x_50 = self.model[50](x_49)
        x_51 = self.model[51](x_50)
        x_52 = self.model[52](x_51)
        x_53 = self.model[53](x_52)
        x_54 = self.model[54](x_37)
        x_55 = self.model[55]([x_54, x_53])
        x_56 = self.model[56](x_55)
        x_57 = self.model[57](x_55)
        x_58 = self.model[58](x_57)
        x_59 = self.model[59](x_58)
        x_60 = self.model[60](x_59)
        x_61 = self.model[61](x_60)
        x_62 = self.model[62]([x_61, x_60, x_59, x_58, x_57, x_56])
        x_63 = self.model[63](x_62)
        x_64 = self.model[64](x_63)
        x_65 = self.model[65](x_64)
        x_66 = self.model[66](x_24)
        x_67 = self.model[67]([x_66, x_65])
        x_68 = self.model[68](x_67)
        x_69 = self.model[69](x_67)
        x_70 = self.model[70](x_69)
        x_71 = self.model[71](x_70)
        x_72 = self.model[72](x_71)
        x_73 = self.model[73](x_72)
        x_74 = self.model[74]([x_73, x_72, x_71, x_70, x_69, x_68])
        x_75 = self.model[75](x_74)
        x_76 = self.model[76](x_75)
        x_77 = self.model[77](x_76)
        x_78 = self.model[78](x_75)
        x_79 = self.model[79](x_78)
        x_80 = self.model[80]([x_79, x_77, x_63])
        x_81 = self.model[81](x_80)
        x_82 = self.model[82](x_80)
        x_83 = self.model[83](x_82)
        x_84 = self.model[84](x_83)
        x_85 = self.model[85](x_84)
        x_86 = self.model[86](x_85)
        x_87 = self.model[87]([x_86, x_85, x_84, x_83, x_82, x_81])
        x_88 = self.model[88](x_87)
        x_89 = self.model[89](x_88)
        x_90 = self.model[90](x_89)
        x_91 = self.model[91](x_88)
        x_92 = self.model[92](x_91)
        x_93 = self.model[93]([x_92, x_90, x_51])
        x_94 = self.model[94](x_93)
        x_95 = self.model[95](x_93)
        x_96 = self.model[96](x_95)
        x_97 = self.model[97](x_96)
        x_98 = self.model[98](x_97)
        x_99 = self.model[99](x_98)
        x_100 = self.model[100]([x_99, x_98, x_97, x_96, x_95, x_94])
        x_101 = self.model[101](x_100)
        x_102 = self.model[102](x_75)
        x_103 = self.model[103](x_88)
        x_104 = self.model[104](x_101)
        x_105 = self.model[105]([x_102, x_103, x_104])
        x_106 = self.model[106](x_75)
        x_107 = self.model[107](x_106)
        x_108 = self.model[108](x_11)
        x_109 = self.model[109]([x_108, x_107])
        x_110 = self.model[110](x_109)
        x_111 = self.model[111](x_110)
        x_112 = self.model[112]([x_75, x_88, x_101])
        x_113 = self.model[113](x_112)
        x_114 = self.model[114](x_113)
        x_115 = self.model[115](x_114)
        x_116 = self.model[116](x_111)
        x_117 = self.model[117]([x_116, x_115])
        x_118 = self.model[118](x_117)
        x_119 = self.model[119](x_118)
        x_120 = self.model[120]([x_75, x_63, x_51])
        x_121 = self.model[121](x_120)
        x_122 = self.model[122](x_121)
        x_123 = self.model[123](x_122)
        x_124 = self.model[124](x_111)
        x_125 = self.model[125]([x_124, x_123])
        x_126 = self.model[126](x_125)
        x_127 = self.model[127](x_126)
        x_128 = self.model[128]([x_105, x_119, x_127])

        return x_128
