import torch
import torch.nn as nn
import torch.nn.functional as F

BN_MOMENTUM = 0.1

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class StageModule(nn.Module):
    def __init__(self, input_branches, output_branches, c):
        """
        构建对应stage，即用来融合不同尺度的实现
        :param input_branches: 输入的分支数，每个分支对应一种尺度
        :param output_branches: 输出的分支数
        :param c: 输入的第一个分支通道数
        """
        super().__init__()
        self.input_branches = input_branches
        self.output_branches = output_branches

        self.branches = nn.ModuleList()
        for i in range(self.input_branches):  # 每个分支上都先通过4个BasicBlock
            w = c * (2 ** i)  # 对应第i个分支的通道数
            branch = nn.Sequential(
                BasicBlock(w, w),
                BasicBlock(w, w),
                BasicBlock(w, w),
                BasicBlock(w, w)
            )
            self.branches.append(branch)

        self.fuse_layers = nn.ModuleList()  # 用于融合每个分支上的输出
        for i in range(self.output_branches):
            self.fuse_layers.append(nn.ModuleList())
            for j in range(self.input_branches):
                if i == j:
                    # 当输入、输出为同一个分支时不做任何处理
                    self.fuse_layers[-1].append(nn.Identity())
                elif i < j:
                    # 当输入分支j大于输出分支i时(即输入分支下采样率大于输出分支下采样率)，
                    # 此时需要对输入分支j进行通道调整以及上采样，方便后续相加
                    self.fuse_layers[-1].append(
                        nn.Sequential(
                            nn.Conv2d(c * (2 ** j), c * (2 ** i), kernel_size=1, stride=1, bias=False),
                            nn.BatchNorm2d(c * (2 ** i), momentum=BN_MOMENTUM),
                            nn.Upsample(scale_factor=2.0 ** (j - i), mode='nearest')
                        )
                    )
                else:  # i > j
                    # 当输入分支j小于输出分支i时(即输入分支下采样率小于输出分支下采样率)，
                    # 此时需要对输入分支j进行通道调整以及下采样，方便后续相加
                    # 注意，这里每次下采样2x都是通过一个3x3卷积层实现的，4x就是两个，8x就是三个，总共i-j个
                    ops = []
                    # 前i-j-1个卷积层不用变通道，只进行下采样
                    for k in range(i - j - 1):
                        ops.append(
                            nn.Sequential(
                                nn.Conv2d(c * (2 ** j), c * (2 ** j), kernel_size=3, stride=2, padding=1, bias=False),
                                nn.BatchNorm2d(c * (2 ** j), momentum=BN_MOMENTUM),
                                nn.ReLU(inplace=True)
                            )
                        )
                    # 最后一个卷积层不仅要调整通道，还要进行下采样
                    ops.append(
                        nn.Sequential(
                            nn.Conv2d(c * (2 ** j), c * (2 ** i), kernel_size=3, stride=2, padding=1, bias=False),
                            nn.BatchNorm2d(c * (2 ** i), momentum=BN_MOMENTUM)
                        )
                    )
                    self.fuse_layers[-1].append(nn.Sequential(*ops))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 每个分支通过对应的block
        x = [branch(xi) for branch, xi in zip(self.branches, x)]

        # 接着融合不同尺寸信息
        x_fused = []
        for i in range(len(self.fuse_layers)):
            x_fused.append(
                self.relu(
                    sum([self.fuse_layers[i][j](x[j]) for j in range(len(self.branches))])
                )
            )

        return x_fused

# ASFF模块
class ASFF(nn.Module):
    def __init__(self, level, base_channel=32, rfb=False, vis=False):
        super(ASFF, self).__init__()
        self.level = level
        self.dim = [base_channel * (2 ** i) for i in range(4)]  # HRNet的4个分支通道
        self.inter_dim = self.dim[self.level]
        self.num_branches = 4

        # 为每个分支都创建通道压缩模块
        self.compress_level_0 = nn.Sequential(
            nn.Conv2d(self.dim[0], self.inter_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.inter_dim, momentum=BN_MOMENTUM)
        )
        self.compress_level_1 = nn.Sequential(
            nn.Conv2d(self.dim[1], self.inter_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.inter_dim, momentum=BN_MOMENTUM)
        )
        self.compress_level_2 = nn.Sequential(
            nn.Conv2d(self.dim[2], self.inter_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.inter_dim, momentum=BN_MOMENTUM)
        )
        self.compress_level_3 = nn.Sequential(
            nn.Conv2d(self.dim[3], self.inter_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.inter_dim, momentum=BN_MOMENTUM)
        )

        # 每个level对应不同的expand输出通道
        expand_out_channels = {
            0: self.inter_dim * 4,
            1: self.inter_dim * 2,
            2: self.inter_dim,
            3: self.inter_dim // 2
        }[level]

        self.expand = nn.Sequential(
            nn.Conv2d(self.inter_dim, expand_out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(expand_out_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=False)
        )

        # 权重压缩后通道数
        compress_c = 8 if rfb else 16

        self.weight_levels = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.inter_dim, compress_c, kernel_size=1, bias=False),
                nn.BatchNorm2d(compress_c, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=False)
            ) for _ in range(self.num_branches)
        ])

        self.weight_fusion = nn.Sequential(
            nn.Conv2d(compress_c * self.num_branches, self.num_branches, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.num_branches, momentum=BN_MOMENTUM)
        )

        self.vis = vis
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x_levels):
        if self.level == 0:
            # level0的target_size
            target_size = x_levels[0].shape[2:]
            # 只有x_levels[0]是原尺寸，其他都up采样
            level_0 = self.compress_level_0(x_levels[0])
            level_1 = F.interpolate(self.compress_level_1(x_levels[1]), size=target_size, mode='bilinear',
                                    align_corners=False)
            level_2 = F.interpolate(self.compress_level_2(x_levels[2]), size=target_size, mode='bilinear',
                                    align_corners=False)
            level_3 = F.interpolate(self.compress_level_3(x_levels[3]), size=target_size, mode='bilinear',
                                    align_corners=False)
        elif self.level == 1:
            target_size = x_levels[1].shape[2:]
            level_0 = F.interpolate(self.compress_level_0(x_levels[0]), size=target_size, mode='bilinear',
                                    align_corners=False)
            level_1 = self.compress_level_1(x_levels[1])
            level_2 = F.interpolate(self.compress_level_2(x_levels[2]), size=target_size, mode='bilinear',
                                    align_corners=False)
            level_3 = F.interpolate(self.compress_level_3(x_levels[3]), size=target_size, mode='bilinear',
                                    align_corners=False)
        elif self.level == 2:
            target_size = x_levels[2].shape[2:]
            level_0 = F.interpolate(self.compress_level_0(x_levels[0]), size=target_size, mode='bilinear',
                                    align_corners=False)
            level_1 = F.interpolate(self.compress_level_1(x_levels[1]), size=target_size, mode='bilinear',
                                    align_corners=False)
            level_2 = self.compress_level_2(x_levels[2])
            level_3 = F.interpolate(self.compress_level_3(x_levels[3]), size=target_size, mode='bilinear',
                                    align_corners=False)
        else:  # self.level == 3
            target_size = x_levels[3].shape[2:]
            level_0 = F.interpolate(self.compress_level_0(x_levels[0]), size=target_size, mode='bilinear',
                                    align_corners=False)
            level_1 = F.interpolate(self.compress_level_1(x_levels[1]), size=target_size, mode='bilinear',
                                    align_corners=False)
            level_2 = F.interpolate(self.compress_level_2(x_levels[2]), size=target_size, mode='bilinear',
                                    align_corners=False)
            level_3 = self.compress_level_3(x_levels[3])

        levels_resized = [level_0, level_1, level_2, level_3]

        # 后面权重计算和融合不变
        levels_weight_v = [self.weight_levels[i](levels_resized[i]) for i in range(self.num_branches)]
        levels_weight_v_cat = torch.cat(levels_weight_v, dim=1)
        levels_weight = self.weight_fusion(levels_weight_v_cat)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = sum(levels_resized[i] * levels_weight[:, i:i + 1, :, :] for i in range(self.num_branches))

        out = self.expand(fused_out_reduced)
        out = self.relu(out)

        if self.vis:
            return out, levels_weight, fused_out_reduced.sum(dim=1)
        else:
            return out




class HighResolutionNet(nn.Module):
    def __init__(self, base_channel: int = 32, num_joints: int = 14):
        super().__init__()
        self.base_channel = base_channel

        # Stem部分
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)

        # Layer1: Bottleneck结构
        self.layer1 = self._make_layer(Bottleneck, 64, base_channel * 4, blocks=4)

        # Transition1生成2个分支
        self.transition1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(base_channel * 4 * Bottleneck.expansion, base_channel, kernel_size=3, stride=1, padding=1,
                          bias=False),
                nn.BatchNorm2d(base_channel, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=False)
            ),
            nn.Sequential(
                nn.Conv2d(base_channel * 4 * Bottleneck.expansion, base_channel * 2, kernel_size=3, stride=2, padding=1,
                          bias=False),
                nn.BatchNorm2d(base_channel * 2, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=False)
            )
        ])

        # Stage2，输入2分支，输出2分支
        self.stage2 = StageModule(input_branches=2, output_branches=2, c=base_channel)

        # Transition2生成3分支
        self.transition2 = nn.ModuleList([
            nn.Identity(),
            nn.Identity(),
            nn.Sequential(
                nn.Conv2d(base_channel * 2, base_channel * 4, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(base_channel * 4, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=False)
            )
        ])

        # Stage3，输入3分支，输出3分支
        self.stage3 = StageModule(input_branches=3, output_branches=3, c=base_channel)

        # Transition3生成4分支
        self.transition3 = nn.ModuleList([
            nn.Identity(),
            nn.Identity(),
            nn.Identity(),
            nn.Sequential(
                nn.Conv2d(base_channel * 4, base_channel * 8, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(base_channel * 8, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=False)
            )
        ])

        # Stage4，输入4分支，输出4分支
        self.stage4 = StageModule(input_branches=4, output_branches=4, c=base_channel)

        # 新增：4个ASFF模块分别融合4个分支
        self.asff_modules = nn.ModuleList([
            ASFF(level=i, base_channel=base_channel) for i in range(4)
        ])

        # 拼接4个融合特征通道数
        self.final_channel = 128*4

        # 任务头
        self.head = nn.Sequential(
            nn.Conv2d(self.final_channel, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, num_joints, kernel_size=1, bias=True)
        )
#########################################################################这个前面加一句（nn.sigmoid）
    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )
        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Stem部分：两次步长为2的卷积 + BN + ReLU，降低分辨率，提升通道
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        # Layer1，使用Bottleneck模块，输出基础特征
        x = self.layer1(x)

        # Transition1，产生2个分支特征
        x_list = [trans(x) for trans in self.transition1]

        # Stage2，多分支模块，2分支输入输出
        x_list = self.stage2(x_list)

        # Transition2，将2分支扩展到3分支
        # 这里注意：前两个分支用Identity，第三个分支从第二个分支下采样得到
        for i in range(len(self.transition2)):
            if i < len(x_list):
                x_list[i] = self.transition2[i](x_list[i])
            else:
                # 新增分支，基于最后一个已存在分支
                x_list.append(self.transition2[i](x_list[-1]))

        # Stage3，3分支输入输出
        x_list = self.stage3(x_list)

        # Transition3，将3分支扩展到4分支
        for i in range(len(self.transition3)):
            if i < len(x_list):
                x_list[i] = self.transition3[i](x_list[i])
            else:
                x_list.append(self.transition3[i](x_list[-1]))

        # Stage4，4分支输入输出
        x_list = self.stage4(x_list)

        # ASFF融合4个分支，得到4个融合特征
        asff_outs = [self.asff_modules[i](x_list) for i in range(4)]


        # 统一尺寸为最高分辨率分支大小（x_list[0]的尺寸）
        target_size = x_list[0].shape[2:]  # (H, W)
        asff_outs_resized = [F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False) for feat in
                             asff_outs]

        # for i, t in enumerate(asff_outs):
        #     print(f"asff_outs[{i}] shape: {t.shape}")

        # 拼接4个融合特征（通道维度）
        fused_feature = torch.cat(asff_outs_resized, dim=1)

        # print("self.final_channel:", self.final_channel)
        # print("fused_feature shape:", fused_feature.shape)

        # 任务头
        out = self.head(fused_feature)

        out = torch.sigmoid(out)  #归一化


        return out


