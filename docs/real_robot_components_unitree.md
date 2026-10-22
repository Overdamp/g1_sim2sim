# ตารางองค์ประกอบฝั่งหุ่นจริง Unitree G1: จาก Scene ถึง Curriculum

อ่านคู่กับ [ตาราง environment สำหรับ simulation](environment_components.md) เอกสารนี้เรียง 8 ส่วนเหมือนกัน แต่แยกสิ่งที่มีบนหุ่นจริงออกจากสิ่งที่มีเฉพาะ training

**ขอบเขต:** อ้างอิง Unitree SDK2 Python revision `814556d15970dd2ecf1c9984e845ca02ab07e206` และ Unitree RL mjlab revision `1425b15f73bd4095f0df53709d7c389c3eb9e790` ใน workspace พร้อมตรวจแหล่งทางการของ Unitree เมื่อ 19 ตุลาคม 2026 ยังไม่ได้เชื่อมต่อหุ่นจริง จึงไม่ได้ยืนยันรุ่นย่อย firmware สิทธิ์ low-level control หรือพฤติกรรม timeout ของเครื่องเป้าหมาย

คำว่า “พบใน SDK” หมายถึงมี message/API หรือ implementation ใน revision ที่ตรวจ ไม่ได้ยืนยันว่าทุก field/service ใช้งานได้เหมือนกันใน G1 ทุกเครื่อง ให้ยืนยันรุ่นและ firmware ก่อนสร้าง hardware profile

## แหล่งอ้างอิงและวิธีอ่าน

| รหัส | แหล่ง Unitree | ใช้อ้างอิงเรื่อง |
| --- | --- | --- |
| U1 | [G1 low-level example](https://github.com/unitreerobotics/unitree_sdk2_python/blob/814556d15970dd2ecf1c9984e845ca02ab07e206/example/g1/low_level/g1_low_level_example.py) | Motor indices, PR/AB, modes, DDS และ CRC |
| U2 | [G1 message definitions](https://github.com/unitreerobotics/unitree_sdk2_python/tree/814556d15970dd2ecf1c9984e845ca02ab07e206/unitree_sdk2py/idl/unitree_hg/msg/dds_) | Fields และชนิดข้อมูล LowState/LowCmd/IMU/Motor |
| U3 | [G1 LocoClient](https://github.com/unitreerobotics/unitree_sdk2_python/blob/814556d15970dd2ecf1c9984e845ca02ab07e206/unitree_sdk2py/g1/loco/g1_loco_client.py) | High-level locomotion methods |
| U4 | [MotionSwitcherClient](https://github.com/unitreerobotics/unitree_sdk2_python/blob/814556d15970dd2ecf1c9984e845ca02ab07e206/unitree_sdk2py/comm/motion_switcher/motion_switcher_client.py) | ตรวจ/ปล่อย motion mode และ return codes |
| U5 | [Unitree RL mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab/tree/1425b15f73bd4095f0df53709d7c389c3eb9e790/deploy) | ตัวอย่าง policy controller, preprocessing, FSM |
| U6 | [Unitree developer documentation](https://support.unitree.com/home/en/developer/) | จุดเริ่มตรวจคู่มือของรุ่น/firmware จริง |

ตารางที่ระบุ “ออกแบบของเรา” เป็นข้อเสนอสำหรับ `g1_sim2sim` ไม่ใช่ behavior ที่ SDK หรือ firmware รับรองให้แล้ว

## 0. จาก training environment ไปสู่ real robot

| ส่วนใน training | ฝั่งหุ่นจริง | ใครรับผิดชอบ |
| --- | --- | --- |
| Scene | ตัวหุ่น พื้น อุปกรณ์ payload และ sensor ที่ติดตั้งจริง | Hardware profile + ผู้จัดการทดลอง |
| Commands | ความเร็ว/พฤติกรรมเป้าหมายจากผู้ใช้หรือ planner | Command source ของเรา หรือ Unitree high-level service |
| Actions | Motor commands ที่ส่งผ่าน SDK | Policy action adapter + Unitree firmware |
| Observations | LowState + state estimation + controller internal state | Observation builder |
| Rewards | Metrics สำหรับวิเคราะห์ผล | Evaluator; โดยปกติไม่มี PPO update ระหว่าง deploy |
| Terminations | จบ trial, controller fault หรือคำสั่งหยุด | Experiment harness + controller FSM + hardware protection |
| Events | เตรียมท่า เริ่ม/หยุด และ disturbance ที่จัดการได้ | ผู้ควบคุม + runtime FSM |
| Curriculum | ลำดับการทดลองจากง่ายไปยาก | Test plan; ไม่ได้ทำให้ policy เรียนเองถ้าไม่ retrain |

**ไม่มี ManagerBasedRlEnv จำลองโลกจริงอยู่ในหุ่นโดยอัตโนมัติ** ใน deployment เราใช้ส่วนที่จำเป็นของ policy pipeline แล้วรับสถานะจริงผ่าน SDK ส่วน simulator reset/reward/curriculum ต้องแปลงเป็นหน้าที่ของระบบทดสอบ

## 1. Scene — หุ่นและสภาพแวดล้อมจริง

### 1.1 ตารางองค์ประกอบ

| ส่วน | บนหุ่นจริงคืออะไร | สิ่งที่ตรวจได้ / ต้องบันทึก | ต่างจาก simulation อย่างไร | ที่มา |
| --- | --- | --- | --- | --- |
| Robot variant | G1 รุ่นย่อยที่ใช้งานจริง | DoF, joints ที่ล็อก, firmware, low-level access | ชื่อ G1 อย่างเดียวไม่พอเลือก model | U1, U6 |
| Bodies/links | ชิ้นส่วนจริงและการประกอบ | รูปร่าง ทิศ joints และ model revision ที่ตรง | CAD/MJCF อาจไม่ตรงสภาพหลังเพิ่มอุปกรณ์ | U5 + hardware inspection |
| Actuators | Motors, transmission และ embedded control | รุ่น motor, limits และโหมดที่เอกสารรุ่นรองรับ | มี delay, friction, heating และ protection ที่ model อาจไม่มี | U1, U2, U6 |
| Joint sensors | State จาก motor/encoder processing | q/dq, zeros, sign, active motor IDs | มี bias/noise/filter ไม่ใช่ค่าจริงสมบูรณ์แบบ | U2 |
| IMU | Orientation และ inertial measurements | Sensor frame, placement, calibration, units | ไม่มี ground-truth root state ให้ใช้ตรง ๆ | U2, U5 |
| Feet/contact | เท้าสัมผัสพื้นจริง | พื้น รองเท้า/วัสดุเท้า การสึก และ slip | LowState schema ที่ตรวจไม่มี foot-force/contact array | U2 |
| Payload | มือ อุปกรณ์ คอมพิวเตอร์/สิ่งของที่เพิ่ม | น้ำหนักและตำแหน่งติดตั้ง | เปลี่ยน COM/inertia จาก training model | ออกแบบของเรา |
| Terrain | พื้นจริง ทางลาด สิ่งกีดขวาง | วัสดุ ความเรียบ ความลาดเอียง | ตั้ง `terrain_type='plane'` ไม่ได้ทำให้พื้นจริงเรียบ | ออกแบบของเรา |
| Cameras/LiDAR | อุปกรณ์ตามรุ่นและชุดติดตั้ง | Device inventory, streams, timestamps | ไม่ได้รวมอยู่ใน LowState และไม่ใช่ actor input ของ baseline นี้ | U2 + ตรวจอุปกรณ์จริง |
| Lighting | แสงในพื้นที่ทดสอบ | บันทึกหากใช้ vision | ไม่มีผลโดยตรงต่อ actor ที่รับแต่ IMU/joints แต่มีผลต่อ vision estimator | ออกแบบของเรา |
| Power/temperature | แบตเตอรี่และอุณหภูมิของระบบ | Telemetry ที่ firmware ส่งจริงและ thresholds ตามคู่มือ | สมรรถนะอาจเปลี่ยนตามสภาพการใช้งาน | U2, U6 |
| Network/compute | เครื่องรัน controller และช่องทางเชื่อมต่อ | Interface, DDS domain, SDK version, latency | กระบวนการรับส่งมี jitter/dropout | U1, U5 |

### 1.2 ข้อมูลใน hardware profile

| Field ที่เสนอ | วิธีได้ข้อมูล | เกณฑ์ก่อนใช้ |
| --- | --- | --- |
| `robot_variant`, `active_joint_names` | ตรวจรุ่นและ hardware documentation | ตรงกับ policy/model |
| `sdk_motor_ids`, `mode_pr` | U1 และตรวจ field semantics ของรุ่น | Mapping ไม่มีซ้ำ/ขาดและโหมด coordinate ตรง |
| `expected_mode_machine` | อ่านสถานะและเทียบกับรุ่นที่รองรับ | ไม่ bypass mismatch ด้วยเลขจากเครื่องอื่น |
| `q_zero_offsets`, `joint_signs` | Calibration/inspection ที่ถูกต้อง | มีบันทึกวิธีและผลตรวจ |
| `q_limits`, `dq_limits`, `tau_limits` | คู่มือ hardware + controller constraints | มีแหล่งที่มา ไม่คัดจาก sample gains |
| `kp`, `kd`, `action_scale`, `default_q` | Policy artifact และการ validate actuator | ตรง training/deploy contract |
| `imu_frame`, `quaternion_order` | U5 + ทดสอบ orientation ที่รู้คำตอบ | ยืนตรง/เอียงแต่ละแกนแล้วผลแปลงถูก |
| `payload`, `firmware`, `network_profile` | บันทึกจากเครื่องเป้าหมาย | ผูกกับ run metadata |

### 1.3 Motor indices จากตัวอย่าง G1 ของ Unitree

| SDK index | Joint ในตัวอย่าง | ข้อควรตรวจ |
| --- | --- | --- |
| 0–5 | Left hip pitch, hip roll, hip yaw, knee, ankle pitch, ankle roll | Ankle slots มีชื่อ B/A เมื่อใช้ AB mode |
| 6–11 | Right hip pitch, hip roll, hip yaw, knee, ankle pitch, ankle roll | ตรวจ sign และ coordinate mode |
| 12 | Waist yaw | ตรวจกับรุ่นจริง |
| 13–14 | Waist roll, waist pitch | ตัวอย่างเตือนว่าใช้ไม่ได้กับบาง configuration ที่ waist ถูกล็อก |
| 15–21 | Left shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw | Wrist pitch/yaw ไม่ใช้ใน G1 23DoF ตามตัวอย่าง |
| 22–28 | Right shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw | Wrist pitch/yaw ไม่ใช้ใน G1 23DoF ตามตัวอย่าง |

LowCmd/LowState มี **35 motor slots** ตาม IDL แต่ตัวอย่างนี้ควบคุม **29 joints** ขนาด array ไม่ใช่หลักฐานว่ามี 35 motors ที่พร้อมใช้งาน ห้ามสั่งทุก slot โดยอนุมานจาก array length [U1](https://github.com/unitreerobotics/unitree_sdk2_python/blob/814556d15970dd2ecf1c9984e845ca02ab07e206/example/g1/low_level/g1_low_level_example.py), [U2](https://github.com/unitreerobotics/unitree_sdk2_python/tree/814556d15970dd2ecf1c9984e845ca02ab07e206/unitree_sdk2py/idl/unitree_hg/msg/dds_)

## 2. Commands — เราต้องการให้หุ่นทำอะไร

### 2.1 เลือกเส้นทางควบคุม

| เส้นทาง | Input จากเรา | ใครควบคุมการเดิน | เหมาะกับโครงการนี้เมื่อไร |
| --- | --- | --- | --- |
| Unitree high-level | เรียก LocoClient เช่น `Move(vx,vy,vyaw)` | Controller/service ของ Unitree | ใช้ศึกษาหรือใช้งาน controller สำเร็จรูป |
| Custom policy low-level | Velocity command → actor → joint targets → LowCmd | Policy/controller ของเรา | เส้นทางหลักสำหรับ policy ที่เรา train |

อย่าส่ง high-level locomotion กับ full-body low-level control พร้อมกันโดยไม่มี ownership/mode design ที่ถูกต้อง การมี API ทั้งสองแบบไม่ได้หมายความว่าทำงานร่วมกันได้โดยอัตโนมัติ

### 2.2 Methods ที่พบใน local LocoClient

| Method | สิ่งที่ wrapper ทำ | ข้อควรเข้าใจ |
| --- | --- | --- |
| `GetFsmId()` | ขอ FSM ID ผ่าน RPC | ตรวจ return code ก่อนใช้ผล |
| `SetVelocity(vx,vy,omega,duration)` | ส่ง velocity และ duration | Firmware/service ต้องรองรับ; ไม่ใช่การส่ง motor targets |
| `Move(...)` | เรียก SetVelocity | Wrapper ใน revision นี้มี spelling `continous_move`; ตรวจ timeout semantics ก่อนใช้ |
| `StopMove()` | ส่ง velocity ศูนย์ | เป็นคำสั่งหยุดเคลื่อนที่ระดับ high-level ไม่ใช่ hardware emergency stop |
| `Damp()` | ขอ FSM ของ damping | ไม่รับประกันการทรงตัว และไม่แทน stop procedure ของรุ่นนั้น |
| `ZeroTorque()` | ขอ FSM zero torque | หุ่นอาจสูญเสียแรงพยุง; ไม่ใช่ท่ายืน |
| `Start()` | ขอ FSM เริ่มทำงาน | ไม่ใช่ startup function ของ custom policy |
| `SwitchToUserCtrl()` / `SwitchToInternalCtrl(...)` | ขอเปลี่ยน control mode | ต้องยืนยันการรองรับและเงื่อนไขของ firmware |

ที่มา: [U3](https://github.com/unitreerobotics/unitree_sdk2_python/blob/814556d15970dd2ecf1c9984e845ca02ab07e206/unitree_sdk2py/g1/loco/g1_loco_client.py) ไม่ใช้เลข FSM จาก wrapper เป็นสัญญาสากลของ G1 ทุกรุ่น

### 2.3 Commands ของ custom policy

| รายการ | แหล่ง | สิ่งที่เรา implement |
| --- | --- | --- |
| vx/vy/wz | Keyboard/gamepad/planner | Units, body frame, range clamp และ ramp |
| Stop command | ผู้ใช้หรือ command timeout | ส่ง zero target velocity เมื่อ state/policy ยังปกติ |
| Enable/rearm | Operator + FSM | ตรวจ readiness แล้วจึงเปิด policy |
| Command freshness | Local monotonic timestamp | Timeout ที่มีผลทดสอบรองรับ |
| Gait phase | Internal controller state | ไม่ใช่ sensor; period/reset/standing mask ต้องตรง training |

## 3. Actions — คำสั่งที่ส่งไปยังหุ่นจริง

### 3.1 LowCmd และ MotorCmd

| Field | ชนิดข้อมูลใน IDL | หน้าที่ | สิ่งที่ต้องยืนยัน |
| --- | --- | --- | --- |
| `mode_pr` | uint8 | เลือก coordinate/control mode | U1 ระบุ PR=0, AB=1; map ankles/waist ให้ตรง |
| `mode_machine` | uint8 | Machine-mode field | เทียบ LowState และ hardware profile |
| `motor_cmd` | Array 35 MotorCmd | คำสั่งราย motor slot | ส่งเฉพาะ mapping ที่รองรับพร้อมกำหนด unused slots |
| `motor_cmd[i].mode` | uint8 | Motor mode | ตัวอย่างใช้ 1 enable, 0 disable; ตรวจรุ่นจริง |
| `q`, `dq` | float32 | Position/velocity target | Nominal joint units rad และ rad/s ตาม coordinate mode |
| `tau` | float32 | Feedforward torque command | ยืนยัน torque semantics/limits ของโหมดและรุ่น |
| `kp`, `kd` | float32 | Position/velocity feedback gains | ต้องตรง actuator/policy contract |
| `reserve` | Reserved fields | ส่วนสงวนของ schema | ไม่ใช้เป็น custom payload |
| `crc` | uint32 | Message integrity | คำนวณด้วย SDK utility/wrapper ที่เข้ากับ schema |

ที่มา field definitions: [U2](https://github.com/unitreerobotics/unitree_sdk2_python/tree/814556d15970dd2ecf1c9984e845ca02ab07e206/unitree_sdk2py/idl/unitree_hg/msg/dds_) ตัว IDL บอกชื่อ/ชนิดข้อมูล แต่ไม่ได้บอก limits, units และทุก firmware behavior ครบ ต้องอ่านคู่มือและตัวอย่างร่วมกัน

### 3.2 Pipeline ของเรา

| ขั้น | หน้าที่ | จุดตรวจ |
| --- | --- | --- |
| ONNX actor | Observation → action | Finite, dimension, normalization |
| Action processing | Clip/scale/offset ตาม contract | ไม่ทำ scale หรือ clip ซ้ำ |
| Joint mapping | Training order → SDK order | q, gains และ limits ใช้ map เดียวกัน |
| Target generation | q/dq/kp/kd/tau | ไม่กระโดดจาก current pose เข้าท่า reference |
| LowCmd serialization | Fill modes, slots, CRC | ตรวจ model/variant และ message integrity |
| Publish | ส่งผ่าน `rt/lowcmd` | Cadence, deadlines, ownership และ failure handling |

สูตร `tau_ff + kp*(q_target-q) + kd*(dq_target-dq)` เป็นแบบจำลองใน simulator bridge และใช้ทำความเข้าใจ control interface ได้ แต่ไม่ใช่หลักฐานว่า firmware จริงไม่มี saturation, filtering หรือ motor dynamics เพิ่มเติม

ตัวอย่าง Python low-level กำหนดรอบ 0.002 s ขณะที่ G1 policy profile ใช้ 0.02 s และ C++ FSM ของฐานใช้ 0.001 s เป็นคนละ loop ไม่ควรนำ 50 Hz ของ policy ไปสรุปเป็น publish rate ที่ hardware ต้องการ

## 4. Observations — หุ่นส่งอะไรกลับมา

### 4.1 LowState envelope

| Field | Schema | ใช้ทำอะไร | ข้อจำกัด |
| --- | --- | --- | --- |
| `version` | uint32[2] | บันทึกข้อมูล version ของ message | อย่าแปลเป็น firmware release string โดยไม่มีเอกสาร |
| `mode_pr`, `mode_machine` | uint8 | ตรวจ mode/variant compatibility | มี field ไม่ได้แปลว่าทุกค่ารองรับ |
| `tick` | uint32 | ตรวจ progression และ diagnostic | IDL ไม่ยืนยันหน่วยหรือ clock origin; ต้องรองรับ wrap/reset |
| `imu_state` | IMUState | Orientation/inertial inputs | ตรวจ frame/units/calibration |
| `motor_state` | MotorState[35] | Joint feedback | มีเฉพาะ active slots ที่ใช้กับรุ่นจริง |
| `wireless_remote` | uint8[40] | Raw remote-controller data | Decode ด้วย layout ที่ตรง SDK ไม่ถือว่าเป็น velocity tensor สำเร็จรูป |
| `reserve`, `crc` | Reserved + uint32 | Schema/integrity | CRC ไม่ใช่ authentication |

### 4.2 IMU และ motor feedback

| Field | ข้อมูลใน schema | ใช้ใน policy/evaluation | สิ่งที่ไม่ควรสรุปเอง |
| --- | --- | --- | --- |
| `imu_state.quaternion` | float32[4] | คำนวณ projected gravity | U5 ใช้ wxyz แต่ต้องตรวจ convention/placement กับเครื่องจริง |
| `imu_state.gyroscope` | float32[3] | Angular velocity | Sensor frame อาจต้องแปลงให้ตรง training body frame |
| `imu_state.accelerometer` | float32[3] | Estimator/diagnostic | ไม่ใช่ base velocity และต้องตรวจ gravity/specific-force convention |
| `imu_state.rpy` | float32[3] | Diagnostics | ไม่แทน quaternion โดยไม่สนลำดับแกน/representation |
| `imu_state.temperature` | int16 | IMU health logging | IDL ไม่ให้ warning/shutdown threshold |
| `motor_state[i].q`, `dq`, `ddq` | float32 | Joint state/diagnostics | ต้องตรวจ availability และ filtering; ddq อาจไม่เหมาะใช้ตรง ๆ |
| `tau_est` | float32 | Estimated torque | ไม่ใช่การรับรองว่ามี torque sensor วัดตรง |
| `temperature` | int16[2] | Motor health logging | สองช่องมีความหมาย/threshold ตามรุ่น ห้ามตั้งชื่อเองจาก IDL |
| `vol` | float32 | Voltage-related telemetry | ไม่ใช่ battery SOC; ต้องยืนยัน units/semantics |
| `mode`, `motorstate`, `sensor` | Status/raw fields | Health/error handling | ต้องมี bit definitions ก่อน decode error |

Field inventory จาก [U2](https://github.com/unitreerobotics/unitree_sdk2_python/tree/814556d15970dd2ecf1c9984e845ca02ab07e206/unitree_sdk2py/idl/unitree_hg/msg/dds_) การแปลง quaternion/joints อ้างอิง U5 `unitree_articulation.h`

### 4.3 แยกข้อมูลรับได้ คำนวณได้ และยังไม่มี

| สิ่งที่ actor/evaluator อยากได้ | มีตรงใน LowState หรือไม่ | วิธีจัดการ |
| --- | --- | --- |
| Joint positions/velocities | มี | Map indices, offsets, signs แล้ว normalize ตาม contract |
| Base angular velocity | มี gyro แต่ต้องตรวจ frame | Transform เมื่อจำเป็น |
| Projected gravity | ไม่มี field ชื่อนี้ | คำนวณจาก quaternion และ gravity direction |
| Command velocity | ไม่มีเป็น command tensor ของเรา | ใช้ command source ภายใน controller |
| Previous action | ไม่มี | เก็บ buffer ที่ตรง training semantics |
| Gait phase | ไม่มี phase encoder ของ policy นี้ | Implement phase sin/cos และ standing mask ให้ตรง |
| Base linear velocity | ไม่มีใน schema นี้ | ใช้ estimator ที่ validate แล้ว หรือตัดจาก actor แล้ว retrain |
| World position | ไม่มีใน schema นี้ | ใช้ localization/external measurement สำหรับ evaluator |
| Foot contact/force/air time | ไม่มีใน schema นี้ | Estimator/additional sensors ที่ระบุชัด; ห้ามแทนด้วย sim truth |
| Terrain height scan | ไม่มี | ต้องมี perception stack เพิ่มหาก policy ต้องใช้ |
| Battery SOC | ไม่ได้อยู่ใน LowState schema ที่ตรวจ | ตรวจ BMS service/topic ที่รุ่นนั้นรองรับ ไม่อนุมานจาก `vol` |

การมี IDL ของ BMS/pressure sensor อยู่ใน SDK ไม่ได้ยืนยันว่าหุ่นรุ่นนั้น publish topic ดังกล่าวหรือมี sensor นั้นติดตั้ง

## 5. Rewards — เปลี่ยนเป็นการวัดผลบนหุ่นจริง

| ตอน train | ฝั่ง real robot | แหล่งข้อมูล | วิธีรายงาน |
| --- | --- | --- | --- |
| Linear velocity tracking reward | Tracking RMSE | Estimator หรือ external reference | ระบุความคลาดเคลื่อนของเครื่องมือวัด |
| Angular tracking reward | Yaw-rate error | Command + gyro ที่ frame ตรง | แยก steady state กับ transition |
| Upright/posture rewards | Tilt และ posture deviation | IMU + joint states | Peak/RMS และช่วงที่เกิด |
| Action smoothness cost | Target variation/rate | Controller logs | เทียบกับ actual q/dq |
| Foot slip/clearance reward | Slip/foot-height evaluation | Camera/mocap/estimator ที่ validate | ห้ามอ้างว่าวัดได้จาก LowState ตรง ๆ |
| Torque-related metrics | Estimated effort/saturation indicators | tau_est และ target logs | ระบุว่าเป็น estimated values |
| Episode survival | Trial completed without specified failures | Harness + operator observation | บันทึก stop/fault/fall เป็นคนละเหตุการณ์ |

โดยปกติ deploy เฉพาะ actor ไม่ deploy critic/PPO เพื่ออัปเดต weights ระหว่างเดิน การเก็บ metrics ไม่ทำให้ policy เรียนรู้เอง หากจะปรับปรุงต้องกลับไปวิเคราะห์/retrain และออก artifact version ใหม่

## 6. Terminations — จบ trial กับหยุดหุ่นเป็นคนละเรื่อง

| เหตุการณ์ | ใน simulation | บนหุ่นจริงที่เราออกแบบ | หลักฐานที่ต้องมี |
| --- | --- | --- | --- |
| ครบเวลาทดลอง | Reset episode | จบ scoring แล้วเข้าสู่ end-of-trial procedure ที่ทดสอบแล้ว | Mode transitions และ final motion |
| Command source หาย | อาจยังสุ่ม command ต่อ | Zero velocity request เมื่อ state/policy ปกติ | Command age และ stop response |
| LowState เก่า/หาย | อาจไม่มีปัญหานี้ใน direct sim | ไม่ infer จาก stale state; เข้าสู่ fault strategy | State age + detection latency |
| Policy NaN/exception | จบ episode/run | ห้ามส่ง invalid targets; latch fault | Fault cause และ output suppression |
| Controller process ถูก kill | Simulator ยังวิ่งได้ | Controller ส่ง fallback เองไม่ได้แล้ว | ต้องตรวจ downstream timeout/protection ของ hardware |
| Tilt/การล้ม | ตั้ง done แล้ว teleport reset | ใช้ procedure ที่เหมาะกับท่าหุ่นจริง | ไม่สั่งกลับ home pose แบบกระโดด |
| Motor health fault | อาจไม่มีใน model | Decode ตามเอกสารแล้วปฏิบัติตาม hardware procedure | Error fields + actions taken |
| Operator stop | Event ใน harness | ช่องทางหยุดที่ผู้ควบคุมใช้ได้จริง | ทดสอบแยกจาก software StopMove |

Zero velocity, damping, zero torque และ hardware emergency stop มีผลต่างกัน ไม่กำหนดให้เป็นคำสั่งทดแทนกัน การเลือก fallback ต้องทดสอบกับรุ่นและวิธีพยุงที่เหมาะสม ไม่ถือว่าเลือก damping แล้วหุ่นจะยืนได้

## 7. Events — เตรียมโหมด ท่าเริ่ม และเปลี่ยนสถานะ

| Training event | ฝั่ง hardware | เจ้าของหน้าที่ | สิ่งที่ต้องตรวจ |
| --- | --- | --- | --- |
| Startup | ตรวจ health/network/contract | Startup validation | State สด, schema และ artifact ตรง |
| Reset root pose | จัดตำแหน่งหุ่นจริง | Operator/procedure | ไม่มี teleport หรือ set qpos บนหุ่น |
| Reset joint pose | Interpolate จาก current pose | FSM + target generator | Target continuity และ limits |
| Domain randomization: mass/COM | เปลี่ยน payload ที่ทราบค่า | Experimental setup | ใช้ค่าจริงบันทึกไว้ ไม่สุ่มโหลดโดยไม่มีแผน |
| Domain randomization: friction | เปลี่ยนพื้นทดสอบที่ควบคุมได้ | Experimental setup | เริ่มจากพื้น nominal ก่อน |
| Sensor noise/bias | Calibration และการวัด noise จริง | State estimation/analysis | นำข้อมูลกลับไปปรับ sim randomization |
| Push event | Disturbance test ที่ออกแบบเฉพาะ | Test procedure | ไม่ใช้แรงผลักมือสุ่มเพื่อเลียนแบบ velocity injection โดยตรง |
| Policy reset | Clear history/phase/hidden state | Controller FSM | Timing และ action transition ถูกต้อง |

### Control ownership ของ Unitree

U1 ใช้ `MotionSwitcherClient.CheckMode()` และ `ReleaseMode()` ก่อนเริ่ม low-level publishing ส่วน U4 แสดงว่า CheckMode คืน `(code, result)` และ result อาจเป็น `None` เมื่อผิดพลาด จึงต้องตรวจ return code ก่อนเข้าถึงข้อมูล

| งาน | สถานะใน Unitree source | งานเพิ่มเติมของเรา |
| --- | --- | --- |
| ตรวจ motion mode | มี RPC client | Bounded retries, timeout และบันทึกผล |
| ขอปล่อยโหมดเดิม | มี ReleaseMode | ยืนยันผลและเงื่อนไข firmware ไม่สรุปว่าส่ง request แล้วสำเร็จ |
| รับ mode_machine | ตัวอย่างอ่านจาก LowState | ตรวจเทียบ expected variant ก่อนส่ง |
| เริ่ม low-level control | ตัวอย่าง publish LowCmd | FSM readiness, target initialization และ fault handling |
| เปลี่ยนกลับโหมดอื่น | มี service/method ตาม revision | ใช้ procedure ที่รองรับ ไม่สลับเองเพราะ packet กลับมา |

เอกสารนี้ไม่ได้ให้รัน motion example บนหุ่นเพื่อทดสอบ network: ตัวอย่างมีการเคลื่อน joints จริง ให้เริ่มจาก read-only telemetry ก่อน [U1](https://github.com/unitreerobotics/unitree_sdk2_python/blob/814556d15970dd2ecf1c9984e845ca02ab07e206/example/g1/low_level/g1_low_level_example.py), [U4](https://github.com/unitreerobotics/unitree_sdk2_python/blob/814556d15970dd2ecf1c9984e845ca02ab07e206/unitree_sdk2py/comm/motion_switcher/motion_switcher_client.py)

## 8. Curriculum — ลำดับทดสอบจริงจากง่ายไปยาก

เป็นแผนทดสอบที่เสนอ ไม่ใช่ automatic training curriculum หรือขั้นตอนรับรองของ Unitree

| Stage | ทำอะไร | ไปขั้นถัดไปเมื่อ |
| --- | --- | --- |
| H00 | ยืนยันรุ่น/firmware/access และ procedure ของผู้ผลิต | Hardware profile มีข้อมูลครบ |
| H01 | Read-only LowState | Field/frame/rate/health checks ผ่าน |
| H02 | ตรวจ joint mapping/actuation ตาม procedure พร้อมระบบพยุง | Index/sign/coordinate/gains ถูกต้อง |
| H03 | FixStand transition | ไม่มี target jump และ operator stop ใช้งานได้ |
| H04 | Policy stand แบบมีระบบป้องกันการล้มที่เหมาะสม | Behavior และ timings ผ่าน; แยกแรงพยุงออกจากผล balance |
| H05 | Stand test ที่ไม่อาศัยแรงพยุงในการทรงตัว | Metrics ตาม test plan ผ่าน |
| H06 | เดินหน้าช้าในช่วงที่ train พร้อม stop | Tracking/stop response ผ่าน |
| H07 | Turning/sideways หลัง train ครอบคลุม | ไม่ใช้คำสั่งนอก training range |
| H08 | เพิ่มพื้น/payload/duration ทีละอย่าง | Baseline ผ่านและวิเคราะห์ failure ได้ |

แต่ละ stage เก็บ trial ID, operator, policy/config hashes, hardware profile, raw telemetry และเหตุผลหยุด หากไม่ผ่านให้กลับไปแก้ sim/contract/controller ไม่เพิ่มความยากเพื่อชดเชยโดยไม่มีข้อมูล

## 9. ไฟล์ที่ต้องเพิ่มในโครงการก่อน hardware deployment

| ไฟล์ที่เสนอ — ยังไม่ได้สร้าง | เนื้อหา | แหล่งข้อมูล |
| --- | --- | --- |
| `configs/robots/g1_<actual_variant>.yaml` | Validated joints/modes/offsets/limits | Hardware + U1/U2 |
| `configs/transport/real.yaml` | Interface/domain/topics ของเครื่องจริง | Network และ SDK setup |
| `configs/controller/hardware_safety.yaml` | Health limits, timeout/fallback procedure | คู่มือรุ่น + ผลทดสอบ |
| `manifests/hardware.yaml` | รุ่น, firmware, payload, calibration versions | เครื่องเป้าหมาย |
| `docs/hardware_setup.md` | ขั้นตอนของรุ่นจริงที่ตรวจแล้ว | U6 + local validation |
| `docs/hardware_test_protocol.md` | Trial stages, stop/recovery และ metrics | Test plan ของเรา |
| `tools/record_lowstate.py` | Read-only telemetry recorder | SDK subscriber |
| `tests/fixtures/hardware/` | Recorded states ที่ไม่ใช่ข้อมูลลับ | Captures สำหรับ parity regression |

**เงื่อนไขก่อนเริ่มเขียน hardware profile:** ต้องทราบรุ่น G1/DoF, firmware และ payload จริง รายละเอียดนี้ยังไม่อยู่ในข้อมูลที่ยืนยันของโครงการ

## 10. สรุปความสัมพันธ์กับเอกสารเดิม

| เอกสาร | อ่านเพื่อ |
| --- | --- |
| [Environment components](environment_components.md) | เข้าใจสิ่งที่กำหนดใน training simulator |
| [Implementation plan](implementation_plan.md) | งาน train/export/controller/DDS และลำดับ validation |
| [Real robot components — ไฟล์นี้](real_robot_components_unitree.md) | รู้ว่า hardware มีข้อมูล/คำสั่งใด และต้องแทน training managers อย่างไร |

ตอนนี้เป็นเอกสารอธิบายและแผน implementation เท่านั้น ไม่มีการเปิด DDS, ปล่อย motion mode, ส่ง LowCmd หรือปรับ config ของหุ่นจริงจากงานนี้
