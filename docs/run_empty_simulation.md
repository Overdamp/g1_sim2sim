# คำสั่งเปิด G1 เปล่าและตรวจ Unitree SDK/DDS

เป้าหมาย: เปิดหุ่น G1 ใน MuJoCo โดยไม่โหลด policy แล้วรับ `LowState` ผ่าน Unitree SDK2 เพื่อทดสอบเส้นทาง **simulator → DDS → Python subscriber**

สถานะ: คำสั่งอ้างอิง source ของ workspace นี้ ยังไม่ได้ build/run simulator และ SDK subscriber ในขั้นตอนจัดทำคู่มือ ไม่ใช่คำสั่งเชื่อมต่อหรือควบคุมหุ่นจริง

## 1. ตำแหน่งที่ใช้

| ส่วน | ตำแหน่ง |
| --- | --- |
| โครงการของเรา | `/home/luke/nvidia/unitree/g1_sim2sim` |
| ฐาน training/deployment | `/home/luke/nvidia/unitree/unitree_rl_mjlab` |
| Integrated simulator | `/home/luke/nvidia/unitree/unitree_rl_mjlab/simulate` |
| Python SDK | `/home/luke/nvidia/unitree/unitree_sdk2_python` |
| C++ SDK | `/home/luke/nvidia/unitree/unitree_sdk2` |

ใช้ integrated simulator ของ `unitree_rl_mjlab` ในคู่มือนี้ ไม่ใช่ standalone `unitree_mujoco` อีกโฟลเดอร์ ซึ่งมี default robot/domain ต่างกัน

## 2. ตรวจ prerequisites

เปิด terminal ใน desktop session ที่แสดงหน้าต่างได้:

```bash
cd /home/luke/nvidia/unitree/unitree_rl_mjlab
cmake --version
c++ --version
printf '%s\n' "$DISPLAY"
```

Simulator นี้เป็น C++ program ต้องมี Unitree SDK2 ที่ CMake หาได้ พร้อม Boost program_options, GLFW, yaml-cpp, fmt และ MuJoCo library ที่ตรงกับ headers ตาม CMakeLists ของฐาน

อ่าน [setup ของฐาน](../../unitree_rl_mjlab/doc/setup_en.md), [README ของฐาน](../../unitree_rl_mjlab/README.md) และ [C++ SDK README](../../unitree_sdk2/README.md) ก่อนติดตั้ง dependencies หากยังไม่มี ไม่จำเป็นต้องเริ่ม train เพื่อเปิด simulator

## 3. Build simulator — ทำครั้งแรกหรือเมื่อแก้ source

```bash
cd /home/luke/nvidia/unitree/unitree_rl_mjlab
cmake -S simulate -B simulate/build
cmake --build simulate/build -j2
```

รัน build ขั้นที่สองหลัง configure สำเร็จเท่านั้น ตรวจ executable:

```bash
ls -l /home/luke/nvidia/unitree/unitree_rl_mjlab/simulate/build/unitree_mujoco
```

ถ้า `find_package(unitree_sdk2)` ล้มเหลว ให้ติดตั้ง/ระบุ CMake package ของ SDK ตามคู่มือต้นทางก่อน ไม่ใช่แก้ด้วยการติดตั้ง Python SDK อย่างเดียว

## 4. ตั้งค่า G1 เปล่า — ไม่ต้องมี gamepad

เปิดไฟล์:

```bash
nano /home/luke/nvidia/unitree/unitree_rl_mjlab/simulate/config.yaml
```

แก้เฉพาะ keys ต่อไปนี้และเก็บ keys อื่นไว้ เพราะ parser ของต้นทางอ่าน joystick parameters ด้วยแม้ไม่ใช้ joystick:

```yaml
robot: "g1"
robot_scene: "src/assets/robots/unitree_g1/xmls/scene_g1.xml"
domain_id: 0
interface: "lo"
use_joystick: 0
print_scene_information: 1
enable_elastic_band: 0
```

| ค่า | เหตุผล |
| --- | --- |
| `robot: g1` | เลือก bridge/message family ให้ตรงหุ่น |
| `scene_g1.xml` | Scene ของ G1 ที่ integrated simulator ใช้ |
| `domain_id: 0` | ให้ตรงกับ baseline controller และ subscriber ในคู่มือนี้ |
| `interface: lo` | รับส่งภายในเครื่องผ่าน loopback |
| `use_joystick: 0` | ไม่บังคับให้มี gamepad เพื่ออ่านสถานะ |
| `enable_elastic_band: 0` | ไม่มีแรงพยุงจำลอง; หุ่นอาจล้มเมื่อไม่มี controller |

ตัวเลข domain 0 ไม่ใช่การเชื่อมต่อหุ่นจริงโดยตัวมันเอง ในขั้นนี้เรา bind กับ `lo` ทั้งสอง process และยังไม่มี command publisher

ไฟล์นี้เป็น config ของ upstream simulator การแก้ `g1_sim2sim/configs/transport/sim.yaml` อย่างเดียว **ยังไม่มีผล** เพราะ custom launcher ของเรายังไม่ได้ implement

## 5. Terminal A — เปิด simulator

```bash
cd /home/luke/nvidia/unitree/unitree_rl_mjlab
./simulate/build/unitree_mujoco
```

หากต้องการระบุ network/domain จาก CLI ซึ่งมีใน source ปัจจุบัน:

```bash
./simulate/build/unitree_mujoco --robot=g1 --network=lo --domain_id=0
```

เลือกคำสั่งใดคำสั่งหนึ่ง ไม่เปิด simulator สองตัวใน domain เดียวกันสำหรับการทดสอบนี้

สิ่งที่ควรเห็น:

- หน้าต่าง MuJoCo และโมเดล G1
- รายชื่อ scene/joints/sensors ใน terminal เมื่อเปิด print scene information
- หุ่นอาจล้มลงพื้น เพราะไม่มี policy หรือ controller สั่งทรงตัว

การล้มในขั้นนี้ไม่ใช่หลักฐานว่า DDS เสีย และการเห็นหุ่นใน viewer ยังไม่ใช่หลักฐานว่า SDK รับข้อมูลได้

ปล่อย simulator ทำงานต่อระหว่างตรวจ subscriber อย่าเปิด `g1_ctrl` หรือ motion examples ในขั้น read-only นี้

## 6. Terminal B — เลือก Python environment ที่มี SDK

ใช้ environment ของ Unitree ที่แยกจาก standalone mjlab ตาม [setup ของโครงการ](setup.md) หากใช้ชื่อ Conda ตามคู่มือฐาน:

```bash
conda activate unitree_rl_mjlab
python --version
python -m pip show unitree_sdk2py
```

หาก environment ของคุณชื่ออื่น ให้ activate environment นั้นแทน หากยังไม่มี environment ให้สร้างตามคู่มือ setup ก่อน อย่าใช้ `sudo pip` ติดตั้งลง system Python

ติดตั้ง Python SDK ลง environment ที่เลือก หากยังไม่ได้ติดตั้ง:

```bash
python -m pip install -e /home/luke/nvidia/unitree/unitree_sdk2_python
```

ตรวจ import:

```bash
python -c 'from unitree_sdk2py.core.channel import ChannelSubscriber; from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_; print("G1 SDK imports OK")'
```

หาก CycloneDDS installation ล้มเหลว ให้ตรวจ [Python SDK README](../../unitree_sdk2_python/README.md) และเวอร์ชัน dependency ของ SDK การติดตั้งอาจต้องเข้าถึง network ไม่เปลี่ยน dependency pins ของ environment อื่นเพื่อแก้ปัญหานี้

## 7. Terminal B — รับสถานะ 10 วินาทีโดยไม่ส่งคำสั่ง

คัดลอกทั้งบล็อกนี้หลังเปิด simulator แล้ว:

```bash
python - <<'PY'
import threading
import time

from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

lock = threading.Lock()
received = 0
latest = None


def on_state(msg):
    global received, latest
    # Copy only the diagnostic fields we need; no motor commands are created.
    snapshot = (
        time.monotonic(),
        int(msg.mode_machine),
        float(msg.motor_state[3].q),
        tuple(float(v) for v in msg.imu_state.gyroscope),
    )
    with lock:
        received += 1
        latest = snapshot


ChannelFactoryInitialize(0, "lo")
subscriber = ChannelSubscriber("rt/lowstate", LowState_)
subscriber.Init(on_state, 10)

print("Listening: domain=0 interface=lo topic=rt/lowstate type=unitree_hg")
started = time.monotonic()
previous_count = 0
try:
    for _ in range(10):
        time.sleep(1.0)
        with lock:
            count, state = received, latest
        arrivals = count - previous_count
        previous_count = count
        if state is None:
            print("Waiting for LowState...", flush=True)
            continue
        received_at, mode_machine, knee_q, gyro = state
        age_ms = 1000.0 * (time.monotonic() - received_at)
        print(
            f"received={count} arrivals_last_interval={arrivals} "
            f"receive_age_ms={age_ms:.1f} mode_machine={mode_machine} "
            f"left_knee_q={knee_q:.3f} gyro={gyro}",
            flush=True,
        )
except KeyboardInterrupt:
    pass

with lock:
    final_count = received
elapsed = time.monotonic() - started
print(f"Total={final_count}, observed average={final_count / elapsed:.1f} msg/s")
if final_count == 0:
    raise SystemExit("No LowState received. Check simulator, domain and interface.")
PY
```

สคริปต์นี้สร้างเฉพาะ subscriber ไม่มี `LowCmd`, `ChannelPublisher` หรือการเรียก motion-mode service

`receive_age_ms` วัดเวลาตั้งแต่ callback รับ packet ล่าสุดบนเครื่องนี้ **ไม่ใช่ sensor-to-controller latency** และไม่พิสูจน์ว่า simulator physics เดินหน้าหาก bridge ส่ง state เดิมซ้ำ อัตราที่แสดงเป็น observed arrival rate ไม่ใช่การรับรอง physics/control frequency

## 8. ROS 2 — อ่าน LowState และส่งคำสั่งขยับใน simulator

ทำขั้นนี้หลัง simulator เปิดอยู่และ subscriber แบบอ่านอย่างเดียวเห็น state แล้ว ใช้ Terminal B ที่ source ROS 2 Humble และ `unitree_hg` overlay แล้ว ตั้ง DDS สำหรับ simulator:

```bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=0
export CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="lo" /></Interfaces></General></Domain></CycloneDDS>'
ros2 topic list -t | grep -E 'lowstate|lowcmd'
```

อ่าน state ครั้งเดียว:

```bash
ros2 topic echo /lowstate unitree_hg/msg/LowState --once
```

ถ้า topic ใน `ros2 topic list -t` ของเครื่องแสดงชื่อโดยไม่มี leading slash ให้ใช้ชื่อตามที่แสดง (`lowstate`) ครับ ตัวอย่าง ROS 2 Unitree HG ใน repo subscribe ชื่อ `lowstate`; simulator SDK2 bridge ใช้ G1 `rt/lowstate`/`rt/lowcmd` และ ROS2 DDS mapping ของ Unitree เชื่อมชื่อเหล่านี้ให้

### ส่งคำสั่งทดสอบแบบจำกัด

สคริปต์ในโครงการจะ hold ตำแหน่งที่อ่านได้จาก 29 motor slots และขยับ **left elbow index 18** เพิ่ม 0.10 rad ใน 2 วินาที แล้วคืนตำแหน่งเดิมใน 2 วินาที ต้องมี `LowState` สดก่อนจึงเริ่มส่ง:

```bash
cd /home/luke/nvidia/unitree/g1_sim2sim
python3.11 scripts/ros2_sim_joint_demo.py
```

รันเฉพาะเมื่อ simulator ใช้ domain 0/interface `lo` และ ROS2 environment ตั้งค่าตามด้านบน สคริปต์ตรวจสองค่านี้ก่อน publish หากรับ state ไม่ได้จะไม่ส่ง `LowCmd`; หาก state เก่ากว่า 100 ms จะหยุด publisher

**นี่เป็น simulator-only smoke test, ไม่ใช่คำสั่งยืนหรือเดิน:** หุ่นอาจยังล้มอยู่ แต่ควรเห็นข้อศอกขยับเล็กน้อยและกลับ หลังจากทดสอบให้กด `Ctrl+C` ปิด node

ข้อจำกัดสำคัญ: simulator bridge ใน checkout นี้ไม่ตรวจ CRC ขณะที่สคริปต์กำหนด `crc=0` จึงห้ามใช้สคริปต์นี้กับหุ่นจริงเด็ดขาด ขั้นนี้ตรวจ ROS2 `unitree_hg` → DDS → simulator actuator path เท่านั้น ไม่ตรวจว่าคำสั่งผ่านเงื่อนไข CRC/mode ของ firmware จริง

ตัวอย่าง `g1_low_level_example` ใน Unitree ROS2 repo ก็ส่งคำสั่งจริง แต่หลัง 3 วินาทีจะเริ่มแกว่งข้อเท้า/ข้อมือ จึงเป็น motion demo ไม่ใช่ stand policy และไม่ได้ใช้ในขั้นทดสอบนี้

## 9. อ่านผลทดสอบ

| สิ่งที่เห็น | สรุปได้ | ขั้นตรวจต่อ |
| --- | --- | --- |
| Import SDK ผ่าน | Python environment มี message/channel modules | เปิด simulator แล้วตรวจการรับจริง |
| `received` เพิ่มทุกช่วง | DDS subscriber รับ packets ได้ต่อเนื่อง | ตรวจ age และค่า state |
| q/gyro เปลี่ยนเมื่อหุ่นเคลื่อน | Physics state ถูกส่งออกมาถึง subscriber | บันทึกผลและเตรียม joint mapping tests |
| q คงที่เมื่อหุ่นหยุด | อาจเป็นพฤติกรรมปกติ | ดู packet arrivals และสถานะ viewer ประกอบ |
| Packet ยังมาแต่ state ค้าง | อาจ pause/stall หรือหุ่นหยุดนิ่ง | ตรวจ simulation time/การเคลื่อน ไม่ใช้ packet rate ตัดสินอย่างเดียว |
| ไม่ได้รับ packets | ยังไม่ผ่าน communication check | ตรวจทั้งสอง process ใช้ domain 0, lo, unitree_hg และ rt/lowstate |

เกณฑ์ผ่านรอบนี้: simulator เปิดได้, subscriber รับต่อเนื่อง และเห็น state เปลี่ยนสัมพันธ์กับการเคลื่อนของโมเดล

ยัง **ไม่ได้** พิสูจน์ว่า LowCmd ส่งถึง motor bridge, CRC/modes ใช้ได้กับ firmware จริง หรือ policy เดินได้ การทดสอบขาส่งคำสั่งต้องทำเป็นขั้นถัดไปใน sim

## 10. ปิดการทดสอบ

1. Subscriber จบเองหลังประมาณ 10 วินาที หรือกด `Ctrl+C`
2. ปิดหน้าต่าง simulator หรือกด `Ctrl+C` ที่ Terminal A
3. บันทึกผลใน [experiments.md](experiments.md): dependency revisions, config, วันเวลา, observed message rate และข้อผิดพลาดที่พบ

ไม่ต้องรันคำสั่งหยุดมอเตอร์เพิ่มเติมจากคู่มือนี้ เพราะไม่มี controller/publisher ส่งคำสั่งอยู่ หากคุณเปิด controller อื่นเอง ให้ใช้ขั้นตอนหยุดของ controller นั้น

## 11. แก้ปัญหาที่พบบ่อย

| อาการ | ตรวจ/แก้ |
| --- | --- |
| ไม่มี executable | Configure/build ให้สำเร็จตามข้อ 3 |
| CMake หา unitree_sdk2 ไม่เจอ | ตรวจ installation และ CMake prefix ของ C++ SDK |
| Linker หา glfw/yaml-cpp/fmt ไม่เจอ | ติดตั้ง development dependencies ตาม upstream build requirements |
| MuJoCo header/library mismatch | ตรวจ library ที่โหลดกับ version ที่ build ไม่ใช้ library คนละรุ่นปน |
| Joystick error | ตั้ง `use_joystick: 0` ใน `unitree_rl_mjlab/simulate/config.yaml` |
| ไม่มีหน้าต่าง/GLFW error | ตรวจ local graphical session, DISPLAY และ OpenGL driver |
| `ModuleNotFoundError: unitree_sdk2py` | Activate environment ที่ติดตั้ง SDK จริง |
| DDS bind error | ตรวจ interface `lo` มีอยู่และ CycloneDDS config ของ environment |
| No LowState | ตรวจ simulator ยังทำงานและใช้ domain/interface/type ตรงกัน |
| หุ่นล้ม | คาดได้เมื่อไม่มี controller; ไม่ใช่ failure ของ read-only test |

## 12. รอบถัดไปหลังผ่าน read-only

| ลำดับ | งาน | ขอบเขต |
| --- | --- | --- |
| 1 | เก็บ raw states และ joint mapping | อ่านอย่างเดียว |
| 2 | ทดสอบ LowCmd/PD ใน simulator แบบกำหนดขอบเขต | ใช้ sim support เมื่อทดสอบ joint ทีละตัว |
| 3 | Baseline FixStand/policy | ใช้ controller ต้นทางและ policy ตัวอย่าง |
| 4 | Custom policy | ผ่าน observation/action parity ก่อน closed-loop |

ดู [แผน implement](implementation_plan.md), [Environment components](environment_components.md) และ [ข้อมูลฝั่งหุ่นจริง](real_robot_components_unitree.md) ประกอบ
