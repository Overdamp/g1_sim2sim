# ตารางทำความเข้าใจ Environment ของ G1: Scene → Curriculum

ฝั่ง hardware ดู [ตารางองค์ประกอบหุ่นจริง Unitree G1](real_robot_components_unitree.md) ซึ่งเรียงหัวข้อ 8 ส่วนให้เทียบกันได้

เอกสารนี้อธิบายองค์ประกอบทั้ง 8 ตามลำดับ โดยเริ่มลงรายละเอียดที่ **Scene** และเชื่อมไปยัง training กับ sim-to-sim ผ่าน Unitree SDK/DDS

**ฐานอ้างอิง:** source ใน `unitree_rl_mjlab` commit `1425b15f73bd4095f0df53709d7c389c3eb9e790` ที่ตรวจใน workspace ใช้ task `Unitree-G1-Flat` และ configuration ที่ pin mjlab 1.2.0 ไม่ใช่ standalone mjlab 1.6.0 ตัวเลขด้านล่างเป็นค่าจาก source ไม่ใช่ผลทดสอบหรือค่าที่รับรองสำหรับหุ่นจริง

ใน Python training class ชื่อ `ManagerBasedRlEnv` และ config คือ `ManagerBasedRlEnvCfg` ส่วน `ManagerBasedRLEnv` พบใน C++ deployment wrapper ของโครงการด้วย ควรแยก runtime ทั้งสองออกจากกันเมื่ออ่านโค้ด

## แผนที่ภาพรวม

| ลำดับ | องค์ประกอบ | คำถามหลัก | ผลลัพธ์ที่ส่งต่อ |
| --- | --- | --- | --- |
| 1 | Scene | โลกมีอะไร และหุ่นมีโครงสร้างอย่างไร? | Entities, terrain และ sensor data |
| 2 | Commands | ต้องการให้หุ่นทำอะไร? | ความเร็วเป้าหมายที่ actor/rewards ใช้ |
| 3 | Actions | output ของ policy กลายเป็นคำสั่งมอเตอร์อย่างไร? | Actuator targets |
| 4 | Observations | policy และ critic ได้รับข้อมูลอะไร? | Input tensors |
| 5 | Rewards | พฤติกรรมไหนควรเกิดมากขึ้นหรือน้อยลง? | Reward สำหรับเรียนรู้ |
| 6 | Terminations | เมื่อไรควรจบการทดลองหนึ่ง episode? | Failure/timeout flags |
| 7 | Events | เมื่อเริ่ม/reset/ระหว่างวิ่ง ต้องเปลี่ยนค่าอะไร? | State และ model parameters ที่เปลี่ยน |
| 8 | Curriculum | เมื่อไรเพิ่มความยาก และเพิ่มส่วนไหน? | Terrain/command difficulty ใหม่ |

ลำดับในตารางเป็นลำดับศึกษา ไม่ใช่ลำดับเรียกทุก manager ภายใน `env.step()` ทั้งหมด โดย policy และ PPO เป็นส่วนที่ทำงานร่วมกับ environment ไม่ใช่สมาชิกใน Scene

## 1. Scene — โลกที่หุ่นใช้เรียนรู้

### 1.1 Scene ประกอบด้วยอะไร

| ส่วน | คืออะไร | ตัวอย่างใน G1 | มีผลต่อการเรียนอย่างไร | ตำแหน่งที่ต้องดู |
| --- | --- | --- | --- | --- |
| Robot entity | ตัวหุ่นที่รวมโครงสร้างและการเชื่อมกับ simulator | `entities['robot'] = get_g1_robot_cfg()` | กำหนดสิ่งที่ policy ควบคุม | G1 `env_cfgs.py` |
| Terrain | พื้นที่หุ่นสัมผัสและเดิน | Flat ใช้ `plane`; rough ใช้ terrain generator | เปลี่ยนการทรงตัว การก้าว และ contact | `scene.terrain` |
| Bodies/links | ชิ้นส่วนที่มี pose, mass และ inertia | pelvis, torso, ขาและแขน | กำหนดพลศาสตร์ของหุ่น | `g1.xml` |
| Joints | ความสัมพันธ์การเคลื่อนที่ระหว่าง links | hip, knee, ankle, waist, shoulder | กำหนด DoF, axes และข้อจำกัดการเคลื่อนไหว | `g1.xml` |
| Actuators | แบบจำลองที่ทำให้ joint เคลื่อนที่ | `BuiltinPositionActuatorCfg` | แปลง position targets เป็นแรงตอบสนอง | `g1_constants.py` |
| Visual geometry | รูปร่างที่เห็นใน viewer | Mesh ของตัวหุ่น | ใช้ตรวจรูปร่าง/ท่าทาง; ไม่ควรสรุป contact จากภาพอย่างเดียว | MJCF และ mesh assets |
| Collision geometry | รูปร่างที่ physics ใช้ตรวจชน | เท้าและ body collision geoms | กำหนดการสัมผัสพื้นและ self-collision | `CollisionCfg`, MJCF |
| Built-in sensors | Sensors ที่ประกาศใน robot model | Gyroscope และ accelerometer ที่ IMU site | เป็นแหล่งข้อมูลสำหรับ observation หรือ diagnostics | `<sensor>` ใน MJCF |
| Managed sensors | Sensors ที่เพิ่มผ่าน configuration | `feet_ground_contact`, `self_collision` | ใช้ reward, critic และการวัดผล | `scene.sensors` |
| Initial state | Reference pose/state ของ entity | `HOME_KEYFRAME` | จุดเริ่มต้นก่อน reset events ปรับค่า | `EntityCfg.init_state` |
| Parallel environments | จำนวนโลกจำลองที่เรียนรู้พร้อมกัน | Base `num_envs=1`; เปลี่ยนตอน train ได้ | เพิ่มจำนวน transitions ต่อ rollout และใช้ GPU memory เพิ่ม | `scene.num_envs` |
| Lights/materials | ส่วนแสดงผลหรือข้อมูลภาพ | `<light>` ใน MJCF | มีผลต่อภาพ; actor locomotion ชุดนี้ไม่ได้รับภาพ | MJCF/renderer |
| Viewer camera | มุมที่คนใช้ดูหุ่น | Track `torso_link`, distance 3 m | ช่วยตรวจพฤติกรรม ไม่ใช่กล้อง observation ของ actor | `cfg.viewer` |

**ข้อควรเข้าใจ:** Scene ไม่ได้มีแสงเป็น field บังคับแบบเดียวกันทุก framework แสงอาจอยู่ใน MJCF ส่วน timestep/solver อยู่ใน `cfg.sim` และกล้องที่คนดูอยู่ใน `cfg.viewer` แม้ทั้งหมดเกี่ยวข้องกับโลกจำลองเดียวกัน

### 1.2 Config ถูกประกอบขึ้นอย่างไร

| ขั้น | Function / file | สิ่งที่เกิดขึ้น |
| --- | --- | --- |
| A | `make_velocity_env_cfg()` | สร้าง base task: terrain generator, terrain scan, manager configs และ simulation settings |
| B | `unitree_g1_rough_env_cfg()` | ใส่ robot G1, อ้าง terrain scan กับ pelvis, เพิ่ม foot/self-contact sensors และปรับค่าให้เข้ากับ G1 |
| C | `unitree_g1_flat_env_cfg()` | เริ่มจาก rough config แล้วเปลี่ยนพื้นเป็น plane ลบ terrain scan/height observations และ terrain curriculum |
| D | Task registration | ผูก train config, play config และ PPO runner เข้ากับ `Unitree-G1-Flat` |
| E | CLI/custom wrapper | Override เช่นจำนวน environments หรือ iterations ก่อนสร้าง runtime environment |

อ่านเฉพาะ base config อาจเห็นค่าที่ไม่ใช่ค่าท้ายสุดของ flat task ต้องตาม override จนครบ และบันทึก resolved config ของ run จริง

### 1.3 ค่าที่พบใน Scene ของ G1

| รายการ | ค่าจาก source | ความหมาย / ข้อควรระวัง |
| --- | --- | --- |
| Entity name | `robot` | ใช้ชื่อนี้อ้างใน actions, observations, events และ rewards |
| Robot asset | `src/assets/robots/unitree_g1/xmls/g1.xml` | Training model; ไม่ใช่ไฟล์เดียวกับ deployment scene โดยอัตโนมัติ |
| Flat terrain | `terrain_type='plane'`, generator `None` | พื้นราบ ไม่มีความขรุขระจาก generator |
| Rough terrain | `ROUGH_TERRAINS_CFG` | ต้องอ่าน generator config ที่ resolve จาก dependency เวอร์ชันนั้น |
| จำนวน environments ตั้งต้น | `1` | เป็น default ของ source ไม่ใช่คำแนะนำจำนวน env สำหรับ training |
| `scene.extent` | `2.0` | เป็นค่าของ SceneCfg; ตรวจ semantics ใน mjlab 1.2.0 ก่อนใช้เป็นระยะห่างหรือขนาดพื้นที่ ห้ามตีความจากชื่ออย่างเดียว |
| Initial root position | `(0, 0, 0.8)` m | ค่าใน HOME_KEYFRAME ก่อน reset/terrain placement ไม่ใช่ COM height เป้าหมายที่ policy ต้องรักษาเสมอ |
| Reference hip pitch / knee / ankle pitch | `-0.1 / 0.3 / -0.2` rad | ส่วนหนึ่งของท่า home; ต้อง export ทุก joint ตามลำดับที่ resolve จริง |
| Initial joint velocity | `0` | Reset events สามารถเปลี่ยนค่าเริ่มต้นได้ |
| Collision preset | `FULL_COLLISION` | เปิด collision geoms ที่กำหนด รวม self-collision |
| Foot collision names | `left/right_foot1...7_collision` | เป็น geometry groups ไม่ใช่ 14 motors หรือ 14 feet |
| Foot friction ใน preset | `0.6` | Startup event สุ่มใหม่ช่วง 0.3–1.6 ใน base training config จึงไม่ใช่ effective friction ทุก env |
| Soft joint limit factor | `0.9` | เป็น soft-limit configuration ไม่ใช่การยืนยันว่า hardware limits ถูกบังคับครบแล้ว |
| Actor camera input | ไม่มีใน task นี้ | การเปลี่ยนแสงจึงไม่ได้เปลี่ยน input tensor ของ actor โดยตรง |

### 1.4 Sensors: มีข้อมูลแล้วใครใช้

| Sensor / derived quantity | อยู่ที่ไหน | ใช้โดย | Flat task ยังมีไหม | Sim-to-real ต้องคิดอะไร |
| --- | --- | --- | --- | --- |
| `robot/imu_ang_vel` | Robot sensor ใน training model | Actor + critic | มี | เทียบ frame/หน่วยกับ LowState gyroscope |
| `robot/imu_lin_vel` | Simulated velocity sensor | Critic | มี | ไม่ใช่ข้อมูลที่ IMU วัดความเร็วเชิงเส้นได้โดยตรง; ไม่ส่ง ground truth ให้ actor |
| Projected gravity | คำนวณจาก orientation | Actor + critic | มี | ต้องตรวจ quaternion ordering และ rotation direction |
| Joint q/dq | Articulation state | Actor + critic | มี | ต้อง map ไป SDK motor IDs และตรวจ sign/offset |
| `terrain_scan` | RayCastSensor ที่ pelvis | Height-scan observations | ถูกลบ | ถ้าจะใช้ rough policy ต้องมีแหล่ง terrain perception ที่ deployment รองรับ |
| `feet_ground_contact` | Contact ระหว่าง ankle subtrees กับ terrain | Critic และ rewards | มี | ไม่สมมติว่า LowState มี foot-force field ที่เทียบกันได้โดยตรง |
| `self_collision` | Contact ภายใน robot subtree | Self-collision reward | มี | ใช้ physics simulation ได้; deployment ต้องออกแบบการตรวจชนแยกหากต้องการ |
| Foot height | ตำแหน่ง foot sites จาก simulator | Critic และ rewards | มี | Ground truth สำหรับการเรียน/วัดผล ไม่ใช่ actor input ในชุดนี้ |

การสร้าง sensor กับการเพิ่ม observation เป็นคนละขั้น: sensor อาจใช้เฉพาะ reward โดยไม่ส่งเข้า policy เลย

### 1.5 Physics ที่ต้องตรวจคู่กับ Scene

| รายการ | ค่าที่พบ | ผลต่อพฤติกรรม |
| --- | --- | --- |
| Physics timestep | `0.005 s` | Physics 200 steps/s ในเวลาจำลอง |
| Action decimation | `4` | ใช้หนึ่ง action ต่อ 4 physics steps; policy step 0.02 s หรือ 50 Hz |
| Solver iterations / line search | `10 / 20` ใน base | ความละเอียดและต้นทุนการแก้ physics; ต้องประเมินเมื่อเปลี่ยน |
| CCD iterations ของ flat | `50` | Override จาก G1 rough ที่ตั้ง 500 |
| `njmax` ของ flat | `300` | Solver allocation/configuration ไม่ใช่จำนวน joints ของหุ่น |
| `contact_sensor_maxmatch` ของ flat | `64` | Capacity ของ contact sensor matching ไม่ใช่จำนวน contact ที่ควรเกิด |
| `nconmax` ของ flat | `None` | ให้ framework จัดการตาม semantics ของเวอร์ชันนั้น ไม่ตีความว่าไม่มี contact |
| Actuator gains / armature | คำนวณใน `g1_constants.py` | เปลี่ยนการตอบสนองแม้ policy ให้ action เท่าเดิม |

ความถี่ตาม timestep เป็นความถี่ใน simulation time ไม่ได้ยืนยันว่าเครื่องรันทัน wall-clock real time ต้องวัดแยกใน sim-to-sim

### 1.6 งาน implement Scene ของเรา

| ลำดับ | งาน | ผลลัพธ์ที่ต้องได้ | ตรวจผ่านเมื่อ |
| --- | --- | --- | --- |
| S01 | ยืนยันรุ่น G1 และ robot model | Robot profile พร้อม source/model hash | ไม่ปน 23 กับ 29 DoF |
| S02 | Dump joint/actuator/body/sensor names | รายงานโมเดล | ชื่อและ dimension ตรงกับที่ configs อ้าง |
| S03 | ทำ training ↔ SDK ↔ simulator joint map | `joint_map.yaml` ที่ validated | ไม่มี IDs ซ้ำหรือ joints หาย และ round-trip ผ่าน |
| S04 | ตรวจ plane, contact geoms และ foot sites | Scene inspection report | เท้าสัมผัสพื้นและไม่มีการชนผิดรูปแบบจาก initialization |
| S05 | ตรวจ reference pose และ reset state | q/root state snapshot | ไม่ใช้การเปลี่ยน root height กลบปัญหา model โดยไม่มีหลักฐาน |
| S06 | ตรวจ actor sensor sources | Observation source table | ทุก actor input มีเส้นทาง deploy ที่อธิบายได้ |
| S07 | เทียบ actuator model/gains/limits | Training/deploy comparison | ไม่ซ้อน PD controller หรือสลับ gain order |
| S08 | บันทึก physics และ timing | Resolved config ต่อ run | รู้ว่าอะไรเหมือน/ต่างจาก deployment simulator |

### 1.7 แบบฝึกหัดเริ่มต้นสำหรับ Scene

| ทดลอง | เปลี่ยนอะไร | สิ่งที่ต้องดู | สิ่งที่ยังสรุปไม่ได้ |
| --- | --- | --- | --- |
| เปิดโมเดล | ใช้ model/config เดิม | joints, axes, feet, sensors | ภาพถูกต้องไม่ได้แปลว่า dynamics ถูกต้อง |
| Zero action | คง reference joint targets | contact และการล้ม/ทรงตัว | ไม่ใช่ policy ที่เรียนรู้แล้ว และไม่ใช่ zero torque |
| ลดจำนวน env | เช่นใช้ 1 env เพื่อ debug | อ่าน state/contacts ง่ายขึ้น | ไม่ได้บอก throughput ตอน train หลายร้อย env |
| เปลี่ยน friction ค่าเดียว | ปิด friction randomization ก่อนเปรียบเทียบ | foot slip และการทรงตัว | ผลที่ดีบนพื้นหนึ่งค่าไม่รับรอง real-world robustness |
| เปลี่ยน viewer light/camera | ไม่เปลี่ยน actor input | ภาพและความสะดวกในการตรวจ | ไม่ใช่ perception randomization สำหรับ actor นี้ |

## 2. Commands — เป้าหมายที่ให้หุ่นทำ

| รายการ | ค่าจากฐาน | หน้าที่ | สิ่งที่ต้องระวัง |
| --- | --- | --- | --- |
| Command name | `twist` | ชื่อกลางที่ observations/rewards อ้างถึง | Deploy YAML ใช้ชื่อ `base_velocity` ได้เมื่อ adapter map ถูกต้อง ไม่ได้หมายความว่า tensor เปลี่ยนความหมาย |
| `lin_vel_x` | Base range `(-1.0, 2.0)` m/s | เดินหน้า/ถอยหลัง | Curriculum override ช่วงอีกครั้ง |
| `lin_vel_y` | Base range `(-1.0, 1.0)` m/s | เดินด้านข้าง | ต้องฝึกก่อนสั่งด้านข้างใน deployment |
| `ang_vel_z` | Base range `(-1.0, 1.0)` rad/s | หมุนรอบแกนตั้ง | แยก yaw rate ออกจาก heading angle |
| Resampling | ทุก `3–8 s` | เปลี่ยนเป้าหมายระหว่าง episode | ไม่ใช่ reset robot |
| Standing fraction | `0.05` | สุ่มบาง env ให้คำสั่งหยุด | เป็น probability ไม่ใช่จำนวน 5% ตายตัวทุกครั้ง |
| Heading command | เปิด, stiffness `0.5` | แปลง heading error เป็นคำสั่งหมุนตาม implementation | เมื่อใช้ keyboard yaw rate ต้องปิด/ปรับ heading logic ให้สอดคล้อง |
| Play flat overrides | vx `[-0.5,1]`, vy `[-0.5,0.5]`, wz `[-0.5,0.5]` | ช่วงคำสั่งสำหรับ play ของต้นทาง | ไม่จำเป็นต้องเท่ากับ train range ทุก stage |

**เริ่มของเรา:** stand ให้ทุก command เป็นศูนย์; walk เริ่ม forward range แคบและมี stop samples โดยปิด curriculum ที่จะขยายช่วงเอง บันทึก body frame, units และ timeout behavior ใน contract

## 3. Actions — จาก policy ไปมอเตอร์

| ขั้น | ข้อมูล / config | หน้าที่ | การตรวจ sim-to-sim |
| --- | --- | --- | --- |
| Network output | Action vector | ตัวเลขที่ policy คำนวณ ไม่ใช่ joint angle โดยตรงเสมอ | ตรวจ dimension และ joint order |
| Action term | `joint_pos`, `JointPositionActionCfg` | เลือก robot และ actuators | Resolve `actuator_names=('.*',)` เป็นรายชื่อจริง |
| Offset | `use_default_offset=True` | ใช้ default joint pose เป็นฐาน | Export reference pose ให้ตรง |
| Scale | G1 override เป็น `G1_ACTION_SCALE` | กำหนดความกว้างของการเปลี่ยน joint target | ไม่ใช้ base scalar 0.25 แทน per-joint scale |
| สูตร G1 scale | `0.25 * effort_limit / stiffness` ต่อ actuator group | ได้ scale ตามกลุ่ม actuator | อ่านค่าที่ resolve แล้ว ไม่คำนวณคนละลำดับ |
| Target | `q_target = default_q + scale * action` | เป้าหมายของ joint controller | เทียบก่อน/หลัง clipping ตาม config จริง |
| Actuation | Position actuator ใน training | คำนวณแรงตอบสนองต่อ target | เทียบกับ `tau + kp*(q_target-q) + kd*(dq_target-dq)` ของ DDS bridge |
| SDK output | Motor fields ของ `LowCmd` | ส่ง q/dq/kp/kd/tau | ตรวจ joint map, modes, CRC, limits และ watchdog |

การ clamp/rate-limit เพิ่มตอน deploy เปลี่ยน closed-loop dynamics จึงต้องรวมในการทดสอบและพิจารณา training contract ด้วย

## 4. Observations — ข้อมูลที่ actor และ critic เห็น

### 4.1 Actor ของ flat task

| ลำดับตาม config | Term | ขนาดสำหรับ G1 29 joints | ความหมาย | แหล่ง deploy |
| --- | --- | --- | --- | --- |
| 1 | `base_ang_vel` | 3 | ความเร็วเชิงมุม | LowState gyro + frame conversion |
| 2 | `projected_gravity` | 3 | ทิศแรงโน้มถ่วงใน body frame | คำนวณจาก orientation |
| 3 | `command` | 3 | vx, vy, wz เป้าหมาย | Command source |
| 4 | `phase` | 2 ตาม phase encoder ของ task | Phase ของ gait, period 0.6 s | Phase generator ที่ตรง training |
| 5 | `joint_pos` | 29 | ตำแหน่งเทียบ reference | Mapped q ลบ reference |
| 6 | `joint_vel` | 29 | Joint velocity relative ตาม term | Mapped dq ตาม semantics ของ term |
| 7 | `actions` | 29 | Action ก่อนหน้า | Controller internal buffer |
| รวม | Flat actor input | 98 | เป็นขนาดตาม profile นี้ | ต้องยืนยัน ONNX input shape ของ artifact จริงอีกครั้ง |

Height scan มีใน base/rough แต่ถูกลบจาก flat ทั้ง actor และ critic อย่าคัด observation layout จาก rough มาโหลด flat policy

### 4.2 การประมวลผลและ critic

| เรื่อง | ค่าจาก config / หลักการ | ข้อควรตรวจ |
| --- | --- | --- |
| Concatenation | `concatenate_terms=True` | ลำดับ term ต้องตรงกับ export |
| History | `history_length=1` | ถ้าเพิ่มต้องเปลี่ยน input shape/reset และ deploy buffer |
| Actor noise | เปิด corruption ตอน train | Noise ไม่ใช่ normalization; play ปิด corruption |
| Gyro noise | Uniform `[-0.2,0.2]` | หน่วยตาม observation ไม่ใช่เปอร์เซ็นต์ |
| Gravity noise | Uniform `[-0.05,0.05]` | ตรวจ sensitivity ต่อ orientation |
| Joint position/velocity noise | `[-0.01,0.01]` / `[-1.5,1.5]` | ไม่เท่ากับ encoder bias event ซึ่งเปลี่ยนคนละส่วน |
| Critic extra state | Base linear velocity, foot height/air time/contact/forces | ใช้ privileged information เพื่อเรียนได้ ไม่ต้อง deploy critic |
| Critic corruption | `False` | แม้ term บางตัวมี noise config ก็ต้องดู group-level enable |
| Normalization | อยู่ใน network/runner/export ตาม artifact | ตรวจว่าทำครั้งเดียว ไม่ทำซ้ำหรือข้าม |

**งานสำคัญ:** เก็บ raw-state fixtures แล้วเทียบ training observation builder กับ deployment builder ทีละ term ก่อนปล่อย closed-loop

## 5. Rewards — คะแนนของพฤติกรรม

ตารางนี้เป็น weights ใน Unitree base task หลังรวม reward ที่ G1 เพิ่ม ต้องอ่าน function implementation และ time scaling ของ framework ก่อนเปรียบเทียบขนาด term

| Reward key | Weight | สิ่งที่ส่งเสริม/ลด | ข้อควรสังเกต |
| --- | --- | --- | --- |
| `track_linear_velocity` | +1.0 | ติดตามความเร็วเชิงเส้น | Zero command ก็ใช้สอนให้หยุดได้ |
| `track_angular_velocity` | +1.0 | ติดตาม yaw rate | อย่าให้ความแม่นของการหมุนกลบการล้ม |
| `body_orientation_l2` | -1.0 | ลด torso tilt ตาม function | ไม่เหมือน orientation termination threshold |
| `pose` | +1.0 | รักษาท่าทางตาม standing/walking/running | G1 กำหนด tolerance ต่างกันในแต่ละ joint |
| `body_ang_vel` | -0.05 | ลดการหมุนลำตัวที่ไม่พึงประสงค์ | อ่าน axes ที่ function ใช้ |
| `angular_momentum` | -0.025 | ลด angular momentum | อาศัย simulated sensor |
| `is_terminated` | -200.0 | ลงโทษ termination ตาม function | ตรวจการแยก timeout ก่อนตีความ episode จบทุกแบบ |
| `joint_acc_l2` | -2.5e-7 | ลด joint acceleration | ค่าตัวเลขเล็กเพราะ raw quantity มีสเกลต่างกัน |
| `joint_pos_limits` | -10.0 | ลดการเข้าใกล้/เกิน soft limits ตาม function | Reward ไม่ใช่ hardware safety limit |
| `action_rate_l2` | -0.05 | ลด action เปลี่ยนเร็ว | มากเกินไปอาจทำให้ตอบสนองช้า |
| `foot_gait` | +0.5 | จังหวะ support/swing ตาม phase | Period 0.6 s, offsets [0,0.5] |
| `foot_clearance` | -1.0 | คุมความสูงเท้าตาม function | Target 0.10 m; ไม่ใช่ตั้ง q ให้ยกเท้าตรง ๆ |
| `foot_slip` | -0.25 | ลดการลื่นของเท้าที่สัมผัสพื้น | ต้องตรวจ contact classification |
| `soft_landing` | -0.001 | ลดการลงเท้ารุนแรง | ต้องดูแรง/contact transient จริง |
| `stand_still` | -1.0 | ลดการขยับข้อต่อเมื่อควรหยุด | Command threshold 0.1 |
| `self_collisions` | -1.0 | ลดการชนตัวเอง | G1 เพิ่ม term นี้ ใช้ force threshold 10 |

**วิธีศึกษา:** เปลี่ยน weight ทีละตัว เก็บค่า raw/weighted term, fall rate และวิดีโอ อย่าเลือก policy จาก total reward สูงสุดเพียงอย่างเดียว

## 6. Terminations — จบ episode เมื่อไร

| Term | Config ที่ตรวจ | ความหมาย | Train กับ deployment ต่างกันอย่างไร |
| --- | --- | --- | --- |
| `time_out` | `time_out=True`; train episode 20 s | จบตามเวลาที่กำหนด เป็น truncation | ไม่ใช่ความล้มเหลว; evaluator จบ trial ได้โดยไม่แปลว่าหุ่น fault |
| `fell_over` | Bad orientation limit 70° | จบเมื่อเอียงเกินเกณฑ์ของ function | Controller ต้องมี fault logic ของตัวเอง ไม่รอ reset แบบ training |
| Play episode length | G1 play ตั้ง `1e9` s | ใช้ดู behavior ต่อเนื่อง | ถ้าประเมิน 30 s ต้องกำหนด horizon เอง |

อย่าตีความว่าหุ่นยังไม่ถึง threshold 70° จึงควบคุมได้อย่างปลอดภัย ขอบเขต training failure กับ runtime fault thresholds เป็นคนละข้อกำหนด

## 7. Events — Startup, reset และ disturbances

| Event | เวลาเรียก | ค่าจากฐาน | ผลที่เกิด | แนวทางเริ่มทดลอง |
| --- | --- | --- | --- | --- |
| `reset_base` | Reset | x/y ±0.5 m, z offset 0, yaw ±3.14 rad | สุ่ม root pose ตาม reset semantics | ตรวจว่า offset ถูกบวกกับ origin/reference อย่างไร |
| `reset_robot_joints` | Reset | Position/velocity offset 0 | เริ่ม joints ที่ค่าอ้างอิง | เพิ่มความหลากหลายภายหลังทีละระดับ |
| `push_robot` | ทุก 5–6 s | สุ่ม linear/angular velocity ตาม ranges | เปลี่ยน velocity จำลอง disturbance | ไม่ใช่แรงผลักคงที่หน่วย N; play ลบ event นี้ |
| `foot_friction` | Startup | Absolute range 0.3–1.6 | เปลี่ยน friction ที่ foot geoms | ถ้าต้องการ nominal friction ต้องปิด event หรือกำหนดช่วงคงที่ |
| `encoder_bias` | Startup | ±0.015 | เพิ่ม bias ในระบบ encoder ตาม implementation | ตรวจว่า observation term ของเวอร์ชันนี้อ่านค่าที่ biased จริงหรือไม่ |
| `base_com` | Startup | แต่ละแกน ±0.05 m | เปลี่ยน torso center of mass | ไม่ใช่เพิ่มน้ำหนักหรือเลื่อน visual mesh |
| `randomize_terrain` | Reset ใน play config | เพิ่มโดย G1 rough/play factory | จัดตำแหน่งบน terrain ตาม function | Flat สืบทอด play config จึงต้องตรวจ behavior เมื่อเป็น plane |

Startup randomization กับ reset randomization ให้ความหลากหลายคนละรูปแบบ อย่าสรุปว่าสุ่มใหม่ทุก episode หาก event เป็น `startup`

## 8. Curriculum — เพิ่มความยาก

| Term / stage | เงื่อนไขที่เห็นใน config | สิ่งที่ปรับ | Flat task |
| --- | --- | --- | --- |
| `terrain_levels` | ใช้ `terrain_levels_vel` | ระดับ terrain ตาม logic ของ function | ถูกลบ |
| `command_vel`, stage 0 | `step=0` | vx [-0.5,1], vy [-0.5,0.5], wz [-1,1] | ยังมีตอน train |
| `command_vel`, stage 1 | `step=5000*24` | vx [-1,2], vy [-1,1] | ยังมีตอน train |
| Play mode | `cfg.curriculum={}` | ไม่เพิ่มความยากขณะดูผล | ปิดทั้งหมด |

`step=5000*24` เป็น threshold ตาม counter ที่ curriculum function ใช้ ต้องอ่าน function ก่อนแปลงเป็น PPO iterations หรือ wall time ไม่ใช่ผลว่าหุ่นเก่งแล้วโดยอัตโนมัติ

**สำหรับ custom standing:** ปิด command curriculum ก่อน ไม่เช่นนั้นช่วงความเร็วอาจถูกแก้แม้ตั้งช่วงเริ่มต้นเป็นศูนย์ ส่วน walking เริ่มด้วย fixed ranges ที่วัดผลได้ แล้วค่อยใส่ curriculum

## 9. เชื่อมองค์ประกอบเป็นหนึ่งการทดลอง

| ส่วน | Standing experiment | Walking experiment | หลักฐานที่เก็บ |
| --- | --- | --- | --- |
| Scene | G1 + plane ไม่มีแรงพยุง | ใช้ scene เดียวกันก่อน | Model/config hashes |
| Commands | ศูนย์ทั้งหมด | Forward ช่วงแคบ + stop | Command time series |
| Actions | Joint targets ตาม contract | Contract เดียวกัน | Raw action และ q targets |
| Observations | Hardware-compatible actor inputs | Layout เดิม | Raw state + observation fixtures |
| Rewards | Balance, posture, zero velocity | Tracking + balance + gait | Reward breakdown |
| Terminations | Time limit + fall | Time limit + fall | Timeout/fall counts แยกกัน |
| Events | ควบคุม randomness ให้เทียบได้ | ค่อยเพิ่ม disturbances | Seed และ sampled parameters |
| Curriculum | ปิดก่อน | เปิดหลัง baseline ผ่าน | Stage และ effective ranges |

## 10. Source map สำหรับอ่านต่อ

ลิงก์เหล่านี้ชี้ sibling dependency ใน workspace ตาม layout ปัจจุบัน จึงอาจเปิดไม่ได้บน GitHub repository ของโครงการนี้เพียง repo เดียว

| Source | ใช้ศึกษา |
| --- | --- |
| [Base velocity config](../../unitree_rl_mjlab/src/tasks/velocity/velocity_env_cfg.py) | Manager configs, SceneCfg, simulation settings |
| [G1 environment overrides](../../unitree_rl_mjlab/src/tasks/velocity/config/g1/env_cfgs.py) | Rough → flat, contact sensors และ play overrides |
| [G1 constants](../../unitree_rl_mjlab/src/assets/robots/unitree_g1/g1_constants.py) | Entity, home pose, actuators, collisions, action scales |
| [Training model](../../unitree_rl_mjlab/src/assets/robots/unitree_g1/xmls/g1.xml) | Bodies, joints, sensors, light |
| [Deployment scene](../../unitree_rl_mjlab/src/assets/robots/unitree_g1/xmls/scene_g1.xml) | Model/sensor layout ของ integrated simulator |
| [Task registration](../../unitree_rl_mjlab/src/tasks/velocity/config/g1/__init__.py) | Task IDs และ train/play factories |
| [Custom MDP functions](../../unitree_rl_mjlab/src/tasks/velocity/mdp) | ฟังก์ชันที่ config อ้าง; บางชื่อยังมาจาก mjlab dependency |
| [Deployment profile](../../unitree_rl_mjlab/deploy/robots/g1/config/policy/velocity/v0/params/deploy.yaml) | Observation/action configuration ฝั่ง controller |

อ่านควบคู่กับ [แผน implement](implementation_plan.md) และ [Architecture](architecture.md) ก่อนเริ่มปรับ custom task
