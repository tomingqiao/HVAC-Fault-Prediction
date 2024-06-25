# AHU

## This is a Python based FDD tool for running fault equations inspired by ASHRAE Guideline 36 for HVAC systems across historical datasets with the Pandas computing library. Word documents are generated programmatically with the Python Docx library.

###### Under the hood of a `FaultCondition` class a method (Python function inside a class) called `apply` looks like this below as an example shown for the fault condition 1 which returns the boolean flag as a Pandas dataframe column (`fc1_flag`) if the fault condition is present:
```python
def apply(self, df: pd.DataFrame) -> pd.DataFrame:
    df['static_check_'] = (
        df[self.duct_static_col] < df[self.duct_static_setpoint_col] - self.duct_static_inches_err_thres)
    df['fan_check_'] = (df[self.supply_vfd_speed_col] >=
                        self.vfd_speed_percent_max - self.vfd_speed_percent_err_thres)

    df["fc1_flag"] = (df['static_check_'] & df['fan_check_']).astype(int)

    return df
```
	
###### A report is generated using the Python docx library from passing data into the `FaultCodeReport` class will output a Word document to a directory containing the following info, currently tested on a months worth of data.
* a description of the fault equation
* a snip of the fault equation as defined by ASHRAE
* a plot of the data created with matplotlib with sublots
* data statistics to show the amount of time that the data contains as well as elapsed in hours and percent of time for when the fault condition is `True` and elapsed time in hours for the fan motor runtime.
* a histagram representing the hour of the day for when the fault equation is `True`.
* sensor summary statistics filtered for when the AHU fan is running
故障方程的描述
由ASHRAE定义的故障方程的一个片段
使用matplotlib创建的带有子图的数据图表
数据统计，以显示数据包含的时间量以及当故障条件为True时经过的小时数和时间百分比，以及风扇电机运行时间的经过小时数。
一个直方图，表示当故障方程为True时一天中的小时数。
针对AHU风扇运行时的传感器汇总统计数据的筛选。

### Example Word Doc Report
![Alt text](/air_handling_unit_fdd/images/fc1_report_screenshot_all.png)

### Get Setup
```bash
$ git clone https://github.com/bbartling/open-fdd.git
$ cd open-fdd
$ pip install -r requirements.txt
$ cd air_handling_unit_fdd
```

### Modify with text editor `run_all_config.py`
* set proper column names in your CSV file 
* threshold params need to be engineering unit specific for Imperial or SI units, see `params` screenshot in the images directory
* input arg for CSV file path is `-i`
* input arg for 'do' is `-d` which represents which fault to 'do'
* tested on Windows 10 and Ubuntu 20 LTS on Python 3.10
* output Word Doc reports will be in the final_report directory
在你的CSV文件中设置正确的列名
阈值参数需要针对英制或公制单位具体化，参见图片目录中的params截图
CSV文件路径的输入参数是 -i
'do' 的输入参数是 -d，它代表要执行的故障条件
在Windows 10和Ubuntu 20 LTS上使用Python 3.10进行了测试
输出的Word文档报告将位于final_report目录中
```python
# 'do' fault 1 and 2 for example
$ python ./run_all.py -i ./ahu_data/MZVAV-1.csv -d 1 2
```

## Fault equation descriptions
* Fault Condition 1: Duct static pressure too low with fan operating near 100% speed
* Fault Condition 2: Mix temperature too low; should be between outside and return air
* Fault Condition 3: Mix temperature too high; should be between outside and return air
* Fault Condition 4: PID hunting; too many operating state changes between AHU modes for heating, economizer, and mechanical cooling
* Fault Condition 5: Supply air temperature too low should be higher than mix air
* Fault Condition 6: OA fraction too low or too high, should equal to design % outdoor air requirement
* Fault Condition 7: Supply air temperature too low in full heating
* Fault Condition 8: Supply air temperature and mix air temperature should be approx equal in economizer mode
* Fault Condition 9: Outside air temperature too high in free cooling without additional mechanical cooling in economizer mode
* Fault Condition 10: Outdoor air temperature and mix air temperature should be approx equal in economizer plus mech cooling mode
* Fault Condition 11: Outside air temperature too low for 100% outdoor air cooling in economizer cooling mode
* Fault Condition 12: Supply air temperature too high; should be less than mix air temperature in economizer plus mech cooling mode
* Fault Condition 13: Supply air temperature too high in full cooling in economizer plus mech cooling mode
* Fault Condition 14: Temperature drop across inactive cooling coil (requires coil leaving temp sensor)
* Fault Condition 14: Temperature rise across inactive heating coil (requires coil leaving temp sensor)
故障条件1：风管静压过低，风扇运行接近100%速度
故障条件2：混合温度过低；应该在室外空气和回风之间
故障条件3：混合温度过高；应该在室外空气和回风之间
故障条件4：PID狩猎；在空气处理单元（AHU）的加热、经济器和机械冷却模式之间操作状态变化过多
故障条件5：供气温度过低，应高于混合空气温度
故障条件6：室外空气比例过低或过高，应等于设计的室外空气需求百分比
故障条件7：全热模式下供气温度过低
故障条件8：经济器模式下供气温度和混合空气温度应大致相等
故障条件9：在经济器模式下，无额外机械冷却的自由冷却时室外空气温度过高
故障条件10：在经济器加机械冷却模式下，室外空气温度和混合空气温度应大致相等
故障条件11：在经济器冷却模式下，室外空气温度过低，不适合100%室外空气冷却
故障条件12：在经济器加机械冷却模式下全冷时供气温度过高，应低于混合空气温度
故障条件13：在经济器加机械冷却模式下全冷时供气温度过高
故障条件14：不活跃冷却盘管处的温度下降（需要盘管出口温度传感器）
故障条件14：不活跃加热盘管处的温度上升（需要盘管出口温度传感器）

###### Note - Fault equations expect a float between 0.0 and 1.0 for a control system analog output that is typically expressed in industry HVAC controls as a percentage between 0 and 100% of command. Examples of a analog output could a heating valve, air damper, or fan VFD speed. For sensor input data these can be either float or integer based. Boolean on or off data for control system binary commands the fault equation expects an integer of 0 for Off and 1 for On. A column in your CSV file needs to be named `Date` with a Pandas readable time stamp tested in the format of `12/22/2022  7:40:00 AM`:



