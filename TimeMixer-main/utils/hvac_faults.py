# -*- coding: utf-8 -*-
"""
基于分类式规则的空调系统故障诊断方法 (修改后)
"""

class HVAC_Detector:
    def __init__(self):
        # 传感器数据初始化
        self.ChWVlvPos = None  # 冷却水阀门开度
        self.DaFanPower = None  # 送风机功率
        self.CCALTemp = None  # 冷却盘管出风温度
        self.DaTemp = None  # 送风温度
        self.EaDmprPos = None  # 排风风阀开度
        self.HCALTemp = None  # 加热盘管出风温度
        self.HWVlvPos = None  # 热水阀门开度
        self.MaTemp = None  # 混合空气温度
        self.OaDmprPos = None  # 新风风阀开度
        self.OaTemp = None  # 新风温度
        self.OaTemp_WS = None  # 气象站新风温度
        self.RaDmprPos = None  # 回风风阀开度
        self.RaFanPower = None  # 回风机功率
        self.RaTemp = None  # 回风温度
        self.ReHeatVlvPos_1 = None  # 再热阀门 1 开度
        self.ReHeatVlvPos_2 = None  # 再热阀门 2 开度
        self.ZoneDaTemp_1 = None  # 区域 1 送风温度
        self.ZoneDaTemp_2 = None  # 区域 2 送风温度
        self.ZoneTemp_1 = None  # 区域 1 温度
        self.ZoneTemp_2 = None  # 区域 2 温度

        # 系统参数设置
        self.OaTemp_HT_set = 20  # 室外空气启动暖房盘管的设定温度
        self.OaTemp_cool_set = 26  # 室外空气启动冷房盘管的设定温度
        self.OaTemp_cool_min_set = 18  # 室外空气启动最小新风量的机械式冷房模式的设定温度
        self.ZoneTemp_set_1 = 24  # 区域 1 温度设定值
        self.ZoneTemp_set_2 = 24  # 区域 2 温度设定值

        # 容许值设置
        self.epsilon_OA = 0.5  # 室外空气温度容许值
        self.epsilon_T = 0.5  # 温度测量误差容许值
        self.epsilon_DaTemp = 1  # 送风温度容许值
        self.epsilon_HWVlvPos = 0.05  # 热水阀门开度容许系数
        self.epsilon_ChWVlvPos = 0.05  # 冷却水阀门开度容许系数
        self.Delta_DaTemp = 2  # 送风机温升
        self.Delta_Tmin = 2  # 回风空气和新风温度的最小温差容许值
        self.epsilon_flow = 0.1  # 与流量有关的误差容许值
        self.EaDmprPos_min = 0.1  # 排风风阀最小开度
        self.OaDmprPos_min = 0.1  # 新风风阀最小开度
        self.RaDmprPos_min = 0.1  # 回风风阀最小开度

    def collect_data(self):
        """
        采集传感器数据 (模拟)
        """
        # 此处需要根据实际情况读取传感器数据
        # ...

    def diagnose_fault(self):
        """
        空调系统故障诊断
        """
        self.collect_data()

        if self.OaTemp < self.OaTemp_HT_set:
            # 模式 1: 暖房模式
            self.diagnose_mode_1()
        elif self.OaTemp_HT_set <= self.OaTemp < self.OaTemp_cool_set:
            # 模式 2: 新风冷却模式
            self.diagnose_mode_2()
        elif self.OaTemp_cool_set <= self.OaTemp < self.OaTemp_cool_min_set:
            # 模式 3: 新风最大，机械制冷
            self.diagnose_mode_3()
        elif self.OaTemp_cool_min_set <= self.OaTemp:
            # 模式 4: 新风最小，机械制冷
            self.diagnose_mode_4()

    def diagnose_mode_1(self):
        """
        模式 1 故障诊断
        """
        print("当前运行模式: 模式 1 (暖房模式)")

        # 新风风阀检查
        if self.OaDmprPos > self.OaDmprPos_min + self.epsilon_flow:
            print("警告: 新风风阀未接近最小开度")

        # 排风风阀检查
        if self.EaDmprPos > self.EaDmprPos_min + self.epsilon_flow:
            print("警告: 排风风阀未接近最小开度")

        # 室外空气温度传感器检查
        if abs(self.OaTemp - self.OaTemp_WS) >= self.epsilon_OA:
            print("故障: 室外空气温度传感器故障")
            return

        # 热水阀门检查
        if self.HWVlvPos <= self.epsilon_HWVlvPos:
            print("故障: 热水阀门故障或热水供应不足")
            return

        # 加热盘管检查
        if self.HCALTemp < self.ZoneTemp_set_1 - self.epsilon_T:
            print("故障: 加热盘管效率低或热水流量不足")
            return

        # 混合空气温度传感器检查
        if (self.DaTemp < self.MaTemp + self.Delta_DaTemp - self.epsilon_T) or \
           (abs(self.RaTemp - self.OaTemp) >= self.Delta_Tmin and \
            abs((self.MaTemp - self.RaTemp) / (self.OaTemp - self.RaTemp) - (self.MaTemp - self.RaTemp) / (self.OaTemp - self.RaTemp)) > self.epsilon_flow):
            print("故障: 混合空气温度传感器故障")
            return

        # 送风温度传感器检查
        if abs(self.HWVlvPos - 1) < self.epsilon_HWVlvPos and \
           abs(self.ZoneTemp_set_1 - self.DaTemp) > self.epsilon_DaTemp:
            print("故障: 送风温度传感器故障")
            return

        # 回风温度传感器检查
        if abs(self.RaTemp - self.OaTemp) >= self.Delta_Tmin and \
           abs((self.MaTemp - self.RaTemp) / (self.OaTemp - self.RaTemp) - (self.MaTemp - self.RaTemp) / (self.OaTemp - self.RaTemp)) > self.epsilon_flow:
            print("故障: 回风温度传感器故障")
            return

        print("系统运行正常")

    def diagnose_mode_2(self):
        """
        模式 2 故障诊断
        """
        print("当前运行模式: 模式 2 (新风冷却模式)")

        # 室外空气温度传感器检查
        if abs(self.OaTemp - self.OaTemp_WS) >= self.epsilon_OA:
            print("故障: 室外空气温度传感器故障")
            return

        # 区域温度设定值检查
        if self.OaTemp > self.ZoneTemp_set_1 + self.epsilon_T:
            print("警告: 区域温度设定值可能过低，需要人工调整")

        # 送风温度传感器检查
        if (self.DaTemp > self.RaTemp + self.epsilon_T) and \
           (abs(self.DaTemp - self.MaTemp) > self.epsilon_T):
            print("故障: 送风温度传感器故障")
            return

        # 回风温度传感器检查
        if self.DaTemp > self.RaTemp + self.epsilon_T:
            print("故障: 回风温度传感器故障")
            return

        # 混合空气温度传感器检查
        if abs(self.DaTemp - self.MaTemp) > self.epsilon_T:
            print("故障: 混合空气温度传感器故障")
            return

        print("系统运行正常")

    def diagnose_mode_3(self):
        """
        模式 3 故障诊断
        """
        print("当前运行模式: 模式 3 (新风最大，机械制冷)")

        # 新风风阀检查
        if self.OaDmprPos < 1 - self.epsilon_flow:
            print("警告: 新风风阀未接近最大开度")

        # 排风风阀检查
        if self.EaDmprPos > self.EaDmprPos_min + self.epsilon_flow:
            print("警告: 排风风阀未接近最小开度")

        # 室外空气温度传感器检查
        if abs(self.OaTemp - self.OaTemp_WS) >= self.epsilon_OA:
            print("故障: 室外空气温度传感器故障")
            return

        # 热水阀门检查
        if self.HWVlvPos <= self.epsilon_HWVlvPos:
            print("警告: 热水阀门未完全关闭")

        # 室外空气温度传感器检查 (热水阀门无输出信号时)
        if (self.OaTemp < self.ZoneTemp_set_1 - self.Delta_DaTemp - self.epsilon_T) and \
           (self.OaTemp > self.OaTemp_cool_min_set + self.epsilon_T) and \
           (abs(self.OaTemp - self.MaTemp) > self.epsilon_T):
            print("故障: 室外空气温度传感器故障")
            return

        # 最小新风量机械制冷模式设定温度检查
        if self.OaTemp > self.OaTemp_cool_min_set + self.epsilon_T:
            print("警告: 最小新风量机械制冷模式设定温度可能设置错误，需要人工检查")

        # 混合空气温度传感器检查
        if abs(self.OaTemp - self.MaTemp) > self.epsilon_T:
            print("故障: 混合空气温度传感器故障")
            return

        # 送风温度传感器检查
        if (self.DaTemp > self.MaTemp + self.Delta_DaTemp + self.epsilon_T) and \
           (self.DaTemp > self.RaTemp - self.Delta_DaTemp + self.epsilon_T) and \
           (abs(self.ChWVlvPos - 1) <= self.epsilon_ChWVlvPos) and \
           (self.DaTemp - self.ZoneTemp_set_1 >= self.epsilon_DaTemp):
            print("故障: 送风温度传感器故障")
            return

        # 最小新风量机械制冷模式设定温度检查 (其他情况)
        if self.DaTemp > self.MaTemp + self.Delta_DaTemp + self.epsilon_T:
            print("警告: 最小新风量机械制冷模式设定温度可能设置错误，需要人工检查")

        # 回风温度传感器检查
        if self.DaTemp > self.RaTemp - self.Delta_DaTemp + self.epsilon_T:
            print("故障: 回风温度传感器故障")
            return

        # 冷却水阀门控制信号检查
        if (abs(self.ChWVlvPos - 1) <= self.epsilon_ChWVlvPos) and \
           (self.DaTemp - self.ZoneTemp_set_1 >= self.epsilon_DaTemp):
            print("故障: 冷却水阀门控制信号故障")
            return

        print("系统运行正常")

    def diagnose_mode_4(self):
        """
        模式 4 故障诊断
        """
        print("当前运行模式: 模式 4 (新风最小，机械制冷)")

        # 新风风阀检查
        if self.OaDmprPos > self.OaDmprPos_min + self.epsilon_flow:
            print("警告: 新风风阀未接近最小开度")

        # 排风风阀检查
        if self.EaDmprPos > self.EaDmprPos_min + self.epsilon_flow:
            print("警告: 排风风阀未接近最小开度")

        # 室外空气温度传感器检查
        if abs(self.OaTemp - self.OaTemp_WS) >= self.epsilon_OA:
            print("故障: 室外空气温度传感器故障")
            return

        # 最小新风量机械制冷模式设定温度检查
        if self.OaTemp < self.OaTemp_cool_min_set - self.epsilon_T:
            print("警告: 最小新风量机械制冷模式设定温度可能设置错误，需要人工检查")

        # 送风温度传感器检查
        if (self.DaTemp > self.MaTemp + self.Delta_DaTemp + self.epsilon_T) and \
           (self.DaTemp > self.RaTemp - self.Delta_DaTemp + self.epsilon_T) and \
           (abs(self.ChWVlvPos - 1) <= self.epsilon_ChWVlvPos) and \
           (self.DaTemp - self.ZoneTemp_set_1 >= self.epsilon_DaTemp):
            print("故障: 送风温度传感器故障")
            return

        # 混合空气温度传感器检查
        if self.DaTemp > self.MaTemp + self.Delta_DaTemp + self.epsilon_T:
            print("故障: 混合空气温度传感器故障")
            return

        # 回风温度传感器检查 (送风温度正常时)
        if self.DaTemp > self.RaTemp - self.Delta_DaTemp:
            print("故障: 回风温度传感器故障")
            return

        # 回风温度传感器检查 (其他情况)
        if (abs(self.RaTemp - self.OaTemp) >= self.Delta_Tmin) and \
           (abs((self.MaTemp - self.RaTemp) / (self.OaTemp - self.RaTemp) - (self.MaTemp - self.RaTemp) / (self.OaTemp - self.RaTemp)) > self.epsilon_flow):
            print("故障: 回风温度传感器故障")
            return

        print("系统运行正常")