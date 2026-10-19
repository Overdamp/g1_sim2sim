> เอกสารอ้างอิงเดิม: เก็บรายละเอียดและตัวอย่างเดิมไว้ ชื่อโครงการ `g1_training` และคำสั่งในไฟล์นี้ไม่ใช่ entry point ของโครงการใหม่ ให้เริ่มที่ [README](../../README.md) และ [แผนหลัก](../implementation_plan.md) ก่อนใช้งาน

> ข้อสรุปเรื่องเปลี่ยนเฉพาะ network interface เป็นเป้าหมายของแผนเดิม ไม่ใช่การรับรอง sim-to-real; ใช้ข้อกำหนด contract, hardware profile และ validation จากแผนหลัก

# แนวทาง Implement: G1 Sim-to-Sim ผ่าน Unitree API → ต่อยอดสู่ Sim-to-Real

เอกสารนี้เป็นแผนงานสำหรับเทรน policy การยืน/เดินของ Unitree G1 (29 DoF) แล้วทดสอบแบบ **sim-to-sim**
โดยให้ controller สั่งหุ่นผ่าน **Unitree SDK2 / DDS ตัวจริง** (`rt/lowcmd`, `rt/lowstate`) ไปยัง simulator
เป้าหมายคือ **controller ตัวเดียวกัน (binary + config เดียวกัน) ต้องย้ายไปหุ่นจริงได้โดยเปลี่ยนแค่ network interface**

อ้างอิงจากโค้ดใน `/home/luke/nvidia/unitree/unitree_rl_mjlab` (commit `1425b15`) ณ 19 ต.ค. 2026
สถานะ: **เป็นแผน ยังไม่ได้ลงมือ** — ข้อเท็จจริงเรื่องโค้ดตรวจจากไฟล์จริงแล้ว แต่ยังไม่ได้ build/รัน

---

## สารบัญ

0. [หลักคิด: ทำไม sim-to-sim แบบนี้ถึงพาไป sim-to-real ได้](#0-หลักคิด)
1. [สถาปัตยกรรมและ "สัญญา" ที่ต้องตรงกัน](#1-สถาปัตยกรรมและสัญญาที่ต้องตรงกัน)
2. [ความต่างระหว่าง sim เทรน กับ sim ทดสอบ (sim gap ที่ตั้งใจให้มี)](#2-ความต่างระหว่าง-sim-เทรน-กับ-sim-ทดสอบ)
3. [Phase 0 — เตรียม environment และ build](#phase-0--เตรียม-environment-และ-build)
4. [Phase 1 — Baseline: รัน policy สำเร็จรูปของ Unitree ผ่าน DDS](#phase-1--baseline-รัน-policy-สำเร็จรูปของ-unitree-ผ่าน-dds)
5. [Phase 2 — เทรน policy ของเราเอง](#phase-2--เทรน-policy-ของเราเอง)
6. [Phase 3 — Export: ONNX + deploy.yaml](#phase-3--export-onnx--deployyaml)
7. [Phase 4 — Parity test: พิสูจน์ว่า controller เห็นโลกเหมือนตอนเทรน](#phase-4--parity-test)
8. [Phase 5 — Sim-to-sim stress test](#phase-5--sim-to-sim-stress-test)
9. [Phase 6 — Safety layer ใน controller](#phase-6--safety-layer-ใน-controller)
10. [Phase 7 — Sim-to-real](#phase-7--sim-to-real)
11. [ตาราง Go/No-Go](#ตาราง-gono-go)
12. [ภาคผนวก: ไฟล์ที่ต้องสร้าง/แก้, คำสั่งที่ใช้บ่อย, ปัญหาที่คาดว่าจะเจอ](#ภาคผนวก)

---

## 0. หลักคิด

```
 Train (GPU, MJWarp, 4096 envs)           ── ไม่ใช่สิ่งที่ไปอยู่บนหุ่น
        │  policy.onnx + deploy.yaml
        ▼
 ┌──────────────────────────────┐
 │  g1_ctrl  (C++ controller)   │  ← ตัวนี้คือ "ของจริง" ที่จะไปรันกับหุ่น
 │  FSM + onnxruntime + SDK2    │
 └──────────────┬───────────────┘
                │ DDS: rt/lowcmd ↓   rt/lowstate ↑   (unitree_hg IDL)
        ┌───────┴────────┐
        ▼                ▼
  unitree_mujoco      หุ่น G1 จริง
  (--network=lo)      (--network=enpXs0)
```

กติกา 4 ข้อที่ทำให้ผล sim-to-sim มีความหมายต่อ sim-to-real:

1. **Controller ห้ามรู้ว่าอยู่ใน sim** — ไม่มี `if (sim)` ในโค้ด ทุกอย่างที่ต่างกันอยู่ใน argument `--network` เท่านั้น
2. **ข้อมูลเข้า controller ต้องมาจาก message ที่หุ่นจริงส่งได้เท่านั้น** — IMU (quat, gyro, acc), `q`, `dq`, `tau_est`, wireless remote ห้ามใช้ข้อมูลที่หุ่นจริงไม่มี เช่น ความเร็วลำตัวจริงหรือตำแหน่งในโลก (simulator ส่ง `frame_pos/frame_vel` ได้ แต่ห้ามให้ policy ใช้)
3. **Simulator ทดสอบต้องเป็นคนละตัวกับที่เทรน** — คนละ MJCF, คนละ timestep, คนละวิธีคิด PD (ดูหัวข้อ 2) เพื่อจับ policy ที่ overfit กับ simulator ตอนเทรน
4. **เครื่องมือวัดผลต้องใช้ได้ทั้ง sim และ real** — logger ที่ subscribe DDS ใช้ตัวเดียวกัน ทำให้เทียบ sim กับ real ได้ทีละสัญญาณ

---

## 1. สถาปัตยกรรมและสัญญาที่ต้องตรงกัน

### 1.1 ส่วนประกอบที่ใช้

| ส่วน | ตำแหน่ง | หน้าที่ |
|---|---|---|
| Training | `unitree_rl_mjlab/` (`Unitree-G1-Flat`) | เทรน PPO บน mjlab 1.2.0 + MuJoCo-Warp 3.5.0 |
| Export | `src/tasks/velocity/rl/runner.py` | export `policy.onnx` + metadata ทุกครั้งที่ save checkpoint |
| Controller | `unitree_rl_mjlab/deploy/robots/g1/` → `g1_ctrl` | FSM, อ่าน `deploy.yaml`, รัน ONNX ที่ 50 Hz, ส่ง `lowcmd` |
| Test simulator | `unitree_rl_mjlab/simulate/` → `unitree_mujoco` | MuJoCo CPU + `unitree_sdk2_bridge.h` แปลง DDS ↔ MuJoCo |
| SDK | `unitree_sdk2` (C++, ติดตั้งที่ `/opt/unitree_robotics`) | DDS channel, IDL `unitree_hg` |
| Monitoring | `unitree_sdk2_python` (ยังไม่ได้ติดตั้ง) | สคริปต์ logger/analysis ที่เราจะเขียน |

> ใช้ `unitree_rl_mjlab/simulate/` ไม่ใช่ `unitree/unitree_mujoco/` เป็นตัวหลัก เพราะ config ตั้ง `robot: g1`, `domain_id: 0`
> ตรงกับ `g1_ctrl` ที่ `ChannelFactory::Init(0, ...)` อยู่แล้ว ส่วน `unitree_mujoco` ตั้ง `go2` + domain 1 (ใช้ได้ถ้าแก้ให้ตรง และใช้เป็น simulator ตัวที่สองสำหรับ cross-check ได้)

### 1.2 สัญญา (contract) ระหว่าง training ↔ controller

ทุกข้อนี้ถ้าผิด หุ่นจะล้มทันที หรือแย่กว่านั้นคือ "ดูเหมือนเดินได้แต่แปลก ๆ":

| สัญญา | ฝั่ง training | ฝั่ง controller | วิธีตรวจ (Phase 4) |
|---|---|---|---|
| ลำดับข้อต่อ | `robot.joint_names` จาก `g1.xml` (left leg 0–5, right leg 6–11, waist 12–14, left arm 15–21, right arm 22–28) | `joint_ids_map` ใน `deploy.yaml` (ตอนนี้ identity 0..28) → index ของ `motor_cmd[]` ใน SDK | เทียบกับ `joint_names` ใน ONNX metadata และลำดับมอเตอร์ G1 ใน SDK |
| ลำดับ + ชื่อ observation | `actor` group: `base_ang_vel, projected_gravity, command, phase, joint_pos, joint_vel, actions` | `observations:` ใน `deploy.yaml` ต้องเรียงเหมือนกัน ใช้ชื่อฟังก์ชัน C++: `base_ang_vel, projected_gravity, velocity_commands, gait_phase, joint_pos_rel, joint_vel_rel, last_action` | ป้อน obs ชุดเดียวกันทั้งสองฝั่งแล้วเทียบ |
| Scale/clip ของ obs | `ObservationTermCfg.scale/clip` | `scale`, `clip` ใน `deploy.yaml` | metadata `observation_terms_scale/clip` |
| Action | `JointPositionAction`: `q_target = default + scale * a` | `actions.JointPositionAction.scale/offset` | metadata `action_scale`, `default_joint_pos` |
| PD gain | `joint_stiffness/damping` ใน metadata | `stiffness/damping` ใน `deploy.yaml` → ส่งเป็น `kp/kd` ใน `lowcmd` | เทียบตัวเลข |
| Timing | `sim.dt 0.005 × decimation 4 = 0.02 s` | `step_dt: 0.02` | วัด jitter จาก DDS log |
| Gait phase | `phase(period=0.6)` ให้ (0,0) เมื่อ ‖cmd‖<0.1 | `gait_phase(period: 0.6)` ทำแบบเดียวกัน แต่นับ phase แบบ global ตั้งแต่ `env->reset()` | ตรวจว่า period และ threshold ตรงกัน |
| Command range | `twist.ranges` ช่วงที่เทรนจริง (ระวัง curriculum) | `commands.base_velocity.ranges` ใช้ clamp ค่าจากจอย | ห้ามกว้างกว่าที่เทรน |
| IMU frame | gyro จาก site `imu_in_pelvis`, gravity จาก quat ของ pelvis | `imu_state.gyroscope/quaternion` (sim: site `imu` ที่จุดกำเนิด pelvis) | ดูหัวข้อ 2 |

**ข้อเท็จจริงสำคัญ:** โค้ด Python ใน repo **ไม่ได้สร้าง `deploy.yaml` ให้** มีแค่ไฟล์ที่ Unitree เขียนไว้สำหรับ policy `v0`
ดังนั้นต้องเขียนสคริปต์สร้าง `deploy.yaml` จาก ONNX metadata เอง (Phase 3) — ห้ามคัดลอกของ v0 แล้วหวังว่าจะตรง

---

## 2. ความต่างระหว่าง sim เทรน กับ sim ทดสอบ

ความต่างเหล่านี้คือ "ข้อสอบ" ที่ policy ต้องผ่าน ต้องรู้ก่อนว่ามีอะไรบ้าง จะได้แยกออกว่าล้มเพราะ policy อ่อน หรือเพราะ contract ผิด

| ประเด็น | Training (mjlab) | Test sim (`simulate/` + `scene_g1.xml`) | ผลที่คาด |
|---|---|---|---|
| Physics engine | MuJoCo-Warp (GPU) | MuJoCo C (CPU) | ต่างเล็กน้อยใน contact/solver |
| Timestep | 0.005 s (200 Hz) | ไม่ได้ระบุใน `scene_g1.xml` → ค่า default MuJoCo 0.002 s (500 Hz) — **ต้องยืนยันตอนรัน** (`print_scene_information: 1`) | PD loop เร็วกว่า |
| Actuator | `BuiltinPositionActuator` (PD ใน MuJoCo, kp/kd จาก armature × 10 Hz) | `<motor>` + bridge คำนวณ `tau = tau_ff + kp(q*-q) + kd(dq*-dq)` จาก sensordata ทุก step | ใกล้หุ่นจริงกว่า (หุ่นจริงทำ PD ในบอร์ดมอเตอร์) |
| Torque limit | `effort_limit` ต่อกลุ่มมอเตอร์ | `ctrlrange` ของ `<motor>` (hip 88/139, ankle 50, …) | ควรใกล้กัน ตรวจข้อมือ/แขน |
| Joint dynamics | armature คำนวณจากสเปกมอเตอร์, ไม่มี frictionloss | `<default><joint damping=0.05 armature=0.01 frictionloss=0.2>` | **frictionloss 0.2 ไม่มีตอนเทรน** — gap ที่ชัดที่สุด |
| IMU | gyro ที่ `imu_in_pelvis` (offset จาก pelvis) | `imu_gyro`/`imu_quat` ที่ site `imu` (จุดกำเนิด pelvis) | gyro ของ rigid body เท่ากันทุกจุด แต่ตำแหน่งจริงต่างกัน — accelerometer จะต่าง |
| Noise | มี `Unoise` ใน obs + encoder bias | ไม่มี noise | sim ทดสอบ "สะอาด" กว่าหุ่นจริง → ต้องเติม noise เองใน Phase 5 |
| Latency | ไม่มี (obs→action ใน step เดียว) | DDS + thread policy แยก + sleep_until | มี delay จริง ~ไม่กี่ ms |
| Command | สุ่มทุก 3–8 วินาที | จอย (analog, ต่อเนื่อง) | เปลี่ยนคำสั่งกลางก้าวบ่อยกว่า |

> Phase 5 จะเพิ่ม gap แบบควบคุมได้ (noise, latency, friction, mass) เพื่อให้ sim ทดสอบ "แย่กว่า" หุ่นจริงเล็กน้อย
> หลักการ: **ผ่าน sim ที่แย่กว่าจริง → มีโอกาสผ่านหุ่นจริง** — ไม่ใช่การรับประกัน

---

## Phase 0 — เตรียม environment และ build

**เป้าหมาย:** ทุกชิ้น build ได้ และ DDS คุยกันบน `lo`

### 0.1 Python env สำหรับเทรน (แยกจาก `~/nvidia/mjlab` ซึ่งเป็น mjlab 1.6.0)

- [ ] เปิด conda (`use_conda`) แล้ว
  ```bash
  conda create -n unitree_rl_mjlab python=3.11 -y
  conda activate unitree_rl_mjlab
  cd ~/nvidia/unitree/unitree_rl_mjlab
  pip install -e .          # ดึง mjlab==1.2.0, mujoco-warp==3.5.0
  python scripts/list_envs.py   # ต้องเห็น Unitree-G1-Flat
  ```
- [ ] ยืนยัน torch เห็น GPU: `python -c "import torch;print(torch.cuda.is_available())"`
- [ ] **ห้าม** `pip install` ลง env ของ `~/nvidia/mjlab` (uv) — สอง framework pin เวอร์ชันต่างกัน

### 0.2 System dependencies (C++)

- [ ] `sudo apt install -y libyaml-cpp-dev libboost-all-dev libeigen3-dev libspdlog-dev libfmt-dev libglfw3-dev zlib1g-dev`
- [ ] ตรวจว่า `unitree_sdk2` ติดตั้งที่ `/opt/unitree_robotics` แล้ว (พบ `include/`, `lib/` ในเครื่องนี้) — CMake ของ `simulate/` อ้าง path นี้
- [ ] ตรวจ CycloneDDS: `deploy/robots/g1/CMakeLists.txt` include `/usr/local/include/ddscxx` และ link `ddsc ddscxx` → ถ้าไม่มี ให้ใช้ชุดที่มากับ `unitree_sdk2/thirdparty` หรือ build cyclonedds ตาม README

### 0.3 Build simulator

- [ ] ```bash
  cd ~/nvidia/unitree/unitree_rl_mjlab/simulate
  mkdir -p build && cd build && cmake .. && make -j8
  ./jstest            # ตรวจจอย /dev/input/js0 (มีในเครื่องนี้)
  ```
- [ ] ตรวจ `simulate/config.yaml`: `robot: "g1"`, `robot_scene: src/assets/robots/unitree_g1/xmls/scene_g1.xml`, `domain_id: 0`, `interface: "lo"`, `use_joystick: 1`, `joystick_type` ตรงกับจอยที่มี

### 0.4 Build controller

- [ ] ```bash
  cd ~/nvidia/unitree/unitree_rl_mjlab/deploy/robots/g1
  mkdir -p build && cd build && cmake .. && make -j8
  ```
- [ ] ได้ไฟล์ `g1_ctrl` และ onnxruntime x64 link ได้ (`ldd g1_ctrl | grep onnx`)

### 0.5 Python SDK สำหรับเครื่องมือวัดผล

- [ ] สร้าง env แยก (เช่น `unitree_tools`, Python 3.10/3.11) แล้ว `pip install -e ~/nvidia/unitree/unitree_sdk2_python`
- [ ] ทดสอบ `python -c "import unitree_sdk2py"` (ตอนนี้ยังไม่มี env ไหน import ได้)
- [ ] ⚠️ ต้องให้ CycloneDDS ของ Python กับ C++ คุยกันได้ (ตรวจด้วย logger ใน Phase 1)

**เกณฑ์ผ่าน Phase 0:** เปิด `unitree_mujoco` แล้วสคริปต์ Python subscribe `rt/lowstate` บน `lo` domain 0 ได้ข้อมูล ~500 Hz

---

## Phase 1 — Baseline: รัน policy สำเร็จรูปของ Unitree ผ่าน DDS

**เป้าหมาย:** พิสูจน์ว่า pipeline (sim ↔ DDS ↔ controller ↔ ONNX) ทำงาน **ก่อน** ใช้ policy ของเราเอง
ถ้าข้ามขั้นนี้ เมื่อหุ่นล้มจะแยกไม่ออกว่าผิดที่ policy หรือที่ pipeline

ใช้ `deploy/robots/g1/config/policy/velocity/v0/` ที่ Unitree ให้มา (มี `exported/policy.onnx` + `params/deploy.yaml`)

- [ ] Terminal 1: `cd simulate/build && ./unitree_mujoco`
- [ ] Terminal 2: `cd deploy/robots/g1/build && ./g1_ctrl --network=lo`
  - ต้องเห็น `Connected to robot.` และไม่มี `Unmatched robot type.` (ตรวจ `mode_machine == 5`)
- [ ] ทำ FSM ตามลำดับด้วยจอย:
  1. `L2 + Up` → **FixStand** (interpolate สู่ท่ายืนใน 2 วินาที)
  2. `R2 + A` → **Velocity** (policy เริ่มทำงาน)
  3. ทดสอบ stick ซ้าย = vx/vy, stick ขวา = wz
  4. `L2 + B` → **Passive** (damping อย่างเดียว)
- [ ] เขียน **DDS logger** (`tools/dds_logger.py`) — เครื่องมือหลักที่จะใช้ตลอดทั้งโปรเจคและบนหุ่นจริง
  - subscribe `rt/lowstate` (`unitree_hg.msg.dds_.LowState_`) และ `rt/lowcmd` (`LowCmd_`)
  - บันทึก timestamp, `imu.quaternion/gyroscope/accelerometer`, `motor_state[i].q/dq/tau_est`, `motor_cmd[i].q/kp/kd/tau`, `wireless_remote`
  - เขียนลง `.npz`/`parquet` แยกไฟล์ต่อ session
- [ ] เขียน `tools/analyze_log.py`: plot ความถี่ของ lowcmd (ต้อง ~50 Hz ที่ค่า target เปลี่ยน), tracking error `q_cmd - q`, ความสูง/การเอียงจาก quat, torque ที่ชน limit

**เกณฑ์ผ่าน Phase 1:**
- ยืนนิ่ง 60 วินาทีด้วยคำสั่งศูนย์โดยไม่ล้ม
- เดินหน้า 0.5 m/s และหมุนได้ตามจอย
- logger บันทึกครบ และ jitter ของ policy step < 2 ms (p99)

> ถ้า policy v0 ไม่ผ่าน → ปัญหาอยู่ที่ environment/pipeline ไม่ใช่ policy ห้ามไป Phase 2 จนกว่าจะแก้

---

## Phase 2 — เทรน policy ของเราเอง

**เป้าหมาย:** ได้ checkpoint ที่ยืน/เดินได้ดีใน simulator ตอนเทรน

> บทเรียนจากรันก่อนหน้าในเครื่องนี้: รันใน `~/nvidia/mjlab` และ `IsaacLab` ใช้ `num_envs` แค่ 8–32
> แม้ 30,000 iterations ก็ยังล้มทุก episode (episode ยาวแค่ ~3 วินาทีจาก 20) — PPO ต้องการข้อมูลมากต่อรอบ

- [ ] เทรน (RTX 3060 6 GB — README ของ `g1_locomotion` บันทึกว่า 4096 envs ใช้ ~2.6 GB ใน mjlab; ต้องวัดซ้ำกับ mjlab 1.2.0):
  ```bash
  conda activate unitree_rl_mjlab
  cd ~/nvidia/unitree/unitree_rl_mjlab
  python scripts/train.py Unitree-G1-Flat --env.scene.num-envs=4096
  ```
  ถ้า VRAM ไม่พอ ลดเป็น 2048 → 1024 (ไม่ควรต่ำกว่า 1024)
- [ ] ดู TensorBoard ระหว่างเทรน: `Train/mean_episode_length` → ใกล้ 1000, `Episode_Termination/fell_over` → ใกล้ 0, `track_linear_velocity` / `track_angular_velocity` สูงขึ้น
- [ ] **ระวัง curriculum `command_vel`:** ช่วงคำสั่งขยายเป็น vx (-1, 2), vy (-1, 1) หลัง 5000 iterations — ถ้าหยุดก่อนนั้น policy รู้จักแค่ vx (-0.5, 1.0) ต้องจดไว้แล้วใช้ช่วงนี้ใน `deploy.yaml`
- [ ] ตรวจด้วย play (ใน simulator ตอนเทรน):
  ```bash
  python scripts/play.py Unitree-G1-Flat --checkpoint_file=logs/rsl_rl/g1_velocity/<run>/model_<N>.pt
  ```
- [ ] บันทึกใน `docs/experiments.md`: run id, iterations, num_envs, ค่า reward สุดท้าย, การเปลี่ยน config

**การปรับ config ที่ควรพิจารณาก่อน/ระหว่างเทรน (เพื่อปิด gap ในหัวข้อ 2):**

- [ ] Domain randomization ของ joint friction/damping/armature — ตรวจว่า `mjlab.envs.mdp.dr` ใน 1.2.0 มีฟังก์ชันอะไร (ปัจจุบันใช้ `geom_friction`, `encoder_bias`, `body_com_offset`) แล้วเพิ่ม term ใน `events` ให้ครอบ frictionloss ~0–0.3
- [ ] Action latency/delay (ถ้า mjlab 1.2.0 รองรับ) 0–20 ms
- [ ] สุ่มมวล (`add_base_mass` แบบ ±) สำหรับของที่ติดตัวหุ่นจริง (แบตเตอรี่, มือ)
- [ ] ทำเป็น task ใหม่ `Unitree-G1-Flat-Robust` ใน `src/tasks/velocity/config/g1/` แทนแก้ของเดิม เพื่อเทียบผลได้

**เกณฑ์ผ่าน Phase 2:** ใน play mode ยืนนิ่ง 60 วินาที, เดินตามคำสั่งทุกช่วงที่จะใช้, ไม่ล้มเมื่อโดน push ตาม `push_robot`

---

## Phase 3 — Export: ONNX + deploy.yaml

**เป้าหมาย:** ได้ policy folder ที่ `g1_ctrl` โหลดได้ และทุกตัวเลขมาจาก run ที่เทรนจริง (ไม่พิมพ์มือ)

### 3.1 จัดไฟล์

- [ ] `VelocityOnPolicyRunner.save()` export `policy.onnx` ไว้ใน run folder อัตโนมัติ (มี `policy.onnx.data` ด้วยถ้าโมเดลใหญ่)
- [ ] สร้าง folder:
  ```
  deploy/robots/g1/config/policy/velocity/v1/
  ├── exported/policy.onnx (+ policy.onnx.data ถ้ามี)
  └── params/deploy.yaml
  ```
- [ ] ⚠️ `param::parser_policy_dir()` จะ **เลือก folder ที่ sort แล้วอยู่ท้ายสุด** ที่มี `exported/` อยู่ → ใส่ `v1` แล้ว controller จะใช้ v1 แทน v0 อัตโนมัติ
  - ตั้งชื่อแบบ sort ได้ถูก เช่น `v01`, `v02` หรือ `2026-10-20_run1` (ระวัง `v10` < `v2` ตามลำดับตัวอักษร)
  - หรือระบุ path ตรงใน `config.yaml` (`policy_dir: config/policy/velocity/v1`) เพื่อไม่ให้สับสน — **แนะนำ**

### 3.2 สคริปต์ `tools/gen_deploy_yaml.py` (ต้องเขียนใหม่)

Input: `policy.onnx` (อ่าน metadata) + `params/env.yaml` ของ run → Output: `deploy.yaml`

- [ ] อ่าน ONNX metadata (`onnx.load(...).metadata_props`) ได้: `joint_names`, `joint_stiffness`, `joint_damping`, `default_joint_pos`, `observation_names`, `observation_terms_scale`, `observation_terms_clip`, `observation_terms_history_length`, `action_scale`, `command_names`
- [ ] **ตรวจ `joint_names`** ว่าตรงกับลำดับมอเตอร์ G1 29-DoF ของ SDK ทีละตัว ถ้าต่าง → สร้าง `joint_ids_map` ที่ถูกต้อง (ห้ามสมมติว่าเป็น identity)
- [ ] map ชื่อ observation training → C++:

  | training (`actor`) | `deploy.yaml` | params |
  |---|---|---|
  | `base_ang_vel` | `base_ang_vel` | — |
  | `projected_gravity` | `projected_gravity` | — |
  | `command` | `velocity_commands` | `command_name: base_velocity` |
  | `phase` | `gait_phase` | `period: 0.6` (อ่านจาก env.yaml) |
  | `joint_pos` | `joint_pos_rel` | — |
  | `joint_vel` | `joint_vel_rel` | — |
  | `actions` | `last_action` | — |

  ถ้าเจอชื่อที่ไม่อยู่ในตาราง → สคริปต์ต้อง **error** ไม่ใช่ข้ามไป (C++ ฝั่ง `observation_manager.h` จะหา function ไม่เจอ)
- [ ] เขียน `step_dt` = `sim.mujoco.timestep × decimation` จาก env.yaml (ต้องได้ 0.02)
- [ ] `commands.base_velocity.ranges` = ช่วงที่ policy เทรนจริงตอน checkpoint นั้น (ดู curriculum) และอาจแคบลงเพื่อความปลอดภัย
- [ ] `actions.JointPositionAction.scale` = `action_scale`, `offset` = `default_joint_pos`
- [ ] `actions.JointPositionAction.clip` = limit ข้อต่อจาก MJCF (ตอนนี้ v0 เป็น `null` — แนะนำใส่ ดู Phase 6)
- [ ] เทียบผลลัพธ์กับ `v0/params/deploy.yaml` แบบ diff เพื่อดูว่าโครงสร้างเหมือนกัน (ตัวเลขต่างได้)

### 3.3 ตรวจ ONNX

- [ ] `tools/check_onnx.py`: ป้อน obs สุ่ม 1000 ชุดเข้า PyTorch actor (จาก checkpoint `.pt`) และ ONNX → max abs diff < 1e-5
- [ ] ตรวจ input dim ของ ONNX = ผลรวมมิติ obs (29-DoF: 3+3+3+2+29+29+29 = 98) — คำนวณจาก config ไม่ hardcode

**เกณฑ์ผ่าน Phase 3:** `g1_ctrl` โหลด v1 ได้ไม่ error, ONNX ตรงกับ PyTorch, ไม่มีค่าใน `deploy.yaml` ที่พิมพ์มือ

---

## Phase 4 — Parity test

**เป้าหมาย:** พิสูจน์ว่า **ค่าที่ controller ป้อนเข้า ONNX = ค่าที่ training env ป้อน** ในสถานะเดียวกัน
นี่คือขั้นที่จับ bug contract ส่วนใหญ่ (สลับลำดับข้อต่อ, เครื่องหมายคำสั่ง, scale ผิด)

### 4.1 Observation dump ใน controller

- [ ] เพิ่มโหมด debug ใน `g1_ctrl` (flag เช่น `--dump-obs=path.csv`) ที่บันทึก obs vector, raw action, processed action ทุก step
  - จุดที่แก้: ระหว่าง `env->step()` ใน `State_RLBase` (`deploy/include/FSM/State_RLBase.h`)
  - ปิดได้ และ default ปิด (ต้องใช้บนหุ่นจริงได้ด้วย)

### 4.2 เทียบทีละ term

- [ ] **Static pose test:** ใน FixStand (หุ่นยืนนิ่ง) อ่าน obs จาก dump แล้วตรวจ:
  - `projected_gravity` ≈ (0, 0, -1)
  - `joint_pos_rel` ≈ 0 (เพราะ FixStand ไปที่ default pose เดียวกัน — ตรวจว่า `qs` ใน `config.yaml` = `default_joint_pos`)
  - `gait_phase` = (0, 0) เมื่อคำสั่งศูนย์
- [ ] **Sign test ของคำสั่ง:** ดัน stick ซ้ายไปหน้า → `velocity_commands[0] > 0`, ไปซ้าย → `[1] > 0`, stick ขวาไปซ้าย → `[2] > 0` (C++ ใช้ `ly`, `-lx`, `-rx`)
- [ ] **Joint order test:** ใน Passive ขยับข้อต่อทีละข้อด้วยมือใน MuJoCo viewer (ctrl+drag) แล้วดูว่า index ใน `joint_pos_rel` ที่เปลี่ยนตรงกับ `joint_names[i]`
- [ ] **Replay test (แข็งแรงที่สุด):** บันทึก obs จาก dump ในช่วง Velocity → ป้อนเข้า PyTorch actor ใน Python → action ต้องตรงกับ raw action ใน dump (diff < 1e-4)
- [ ] **Closed-loop A/B:** สั่งคำสั่งเดียวกัน (เช่น vx=0.5 คงที่ 10 วินาที) ใน play (training sim) และใน sim-to-sim → เทียบ joint trajectory, ความถี่การก้าว, ความสูง base ต้องใกล้กัน (ไม่ต้องเท่า)

### 4.3 ตรวจ timing

- [ ] จาก DDS log: ช่วงห่างระหว่าง lowcmd ที่ target เปลี่ยน ≈ 20 ms, p99 jitter < 2 ms
- [ ] วัด latency: เวลาที่ `lowstate` ใหม่เข้ามา → `lowcmd` ใหม่ออกไป (ใช้ timestamp ใน logger)

**เกณฑ์ผ่าน Phase 4:** ทุก test ผ่าน และเขียนไว้เป็นสคริปต์ที่รันซ้ำได้ (จะรันอีกครั้งทุกครั้งที่ได้ policy ใหม่)

---

## Phase 5 — Sim-to-sim stress test

**เป้าหมาย:** ทำให้ simulator ทดสอบ "ยากกว่าหุ่นจริงเล็กน้อย" แล้ว policy ยังผ่าน

ทุกการเปลี่ยนแปลงต้องทำที่ **simulator หรือ MJCF เท่านั้น** ไม่แตะ controller

### 5.1 เครื่องมือที่ต้องเพิ่มใน simulator

- [ ] **Sensor noise ใน bridge** (`simulate/src/unitree_sdk2_bridge.h`): gyro, quat, `q`, `dq` — ใช้ค่าใกล้กับ `Unoise` ตอนเทรน (gyro ±0.2, joint pos ±0.01, joint vel ±1.5)
- [ ] **Latency injection:** buffer `lowcmd` ใน bridge ก่อนใช้ (0, 5, 10, 20 ms)
- [ ] **Perturbation:** ใช้ elastic band (`enable_elastic_band`) หรือเพิ่ม `xfrc_applied` แบบ impulse ด้วยปุ่ม/ตารางเวลา
- [ ] **ตัวแปร MJCF:** สร้างชุด `scene_g1_*.xml` สำหรับ: friction พื้น (0.4/1.0), frictionloss ข้อต่อ (0/0.2/0.4), มวล torso +3 kg, payload ที่มือ
- [ ] **Cross-simulator:** รันกับ `unitree/unitree_mujoco` (แก้ `robot: g1`, scene `g1_29dof.xml`, `domain_id: 0`) เป็น simulator ตัวที่สอง

### 5.2 ชุดทดสอบ (ทำเป็น script อัตโนมัติถ้าทำได้)

ทำ "command script" ด้วย Python ที่ publish `wireless_remote` เอง หรือบันทึกการกดจอยแล้ว replay เพื่อให้ทดสอบซ้ำได้ —
ถ้าทำไม่ได้ ใช้จอยตามลำดับที่กำหนดและบันทึกไว้

| # | Scenario | ระยะเวลา | เกณฑ์ผ่าน |
|---|---|---|---|
| S1 | ยืนนิ่ง cmd = 0 | 120 วินาที | ไม่ล้ม, base drift < 0.3 m, ไม่ขยับเท้า |
| S2 | เดินหน้า 0.3 / 0.6 / 1.0 m/s | 20 วินาที/ระดับ | ความเร็วเฉลี่ยจริง ±20% ของคำสั่ง |
| S3 | ถอยหลัง / เดินข้าง / หมุน | 20 วินาที/แบบ | ไม่ล้ม, ทิศถูก |
| S4 | เปลี่ยนคำสั่งกะทันหัน 0 → 1.0 → 0 | 10 รอบ | ไม่ล้ม, หยุดนิ่งภายใน 2 วินาที |
| S5 | Push ด้านหน้า/ข้าง (impulse) | 10 ครั้ง | ฟื้นตัวได้ ≥ 8/10 |
| S6 | Noise + latency 10 ms | ทำ S1–S4 ซ้ำ | ผ่านทั้งหมด |
| S7 | Friction 0.4, frictionloss 0.4, +3 kg | ทำ S1–S4 ซ้ำ | ผ่านทั้งหมด |
| S8 | FSM transitions: Passive ↔ FixStand ↔ Velocity ซ้ำ ๆ | 20 รอบ | ไม่กระชาก, ไม่ล้มตอนเข้า Velocity |
| S9 | ใช้งานต่อเนื่อง | 15 นาที | ไม่มี memory leak / jitter เพิ่ม |

### 5.3 ตัวชี้วัดจาก DDS log (ใช้ซ้ำกับหุ่นจริงได้)

- [ ] Torque saturation: % ของเวลาที่ `|tau_est|` > 90% ของ limit ต่อข้อ (ควร < 5%)
- [ ] Joint limit proximity: เวลาที่ใกล้ limit < 0.05 rad
- [ ] Action smoothness: `‖a_t − a_{t−1}‖` (สูง = สั่น — หุ่นจริงจะร้อน/เสียง)
- [ ] Tracking error `q_cmd − q` ต่อข้อ
- [ ] Orientation: roll/pitch สูงสุดจาก quaternion

**เกณฑ์ผ่าน Phase 5:** ผ่าน S1–S9 ทั้งหมดใน simulator หลัก และอย่างน้อย S1–S4 ใน simulator ตัวที่สอง
ถ้าไม่ผ่าน → กลับไป Phase 2 ปรับ DR/reward ตามสาเหตุ (ห้าม "แก้" ด้วยการทำให้ simulator ง่ายลง)

---

## Phase 6 — Safety layer ใน controller

**เป้าหมาย:** controller ต้องหยุดได้เองเมื่อมีปัญหา เพราะบนหุ่นจริงความผิดพลาดมีค่าใช้จ่าย

> ⚠️ **ข้อเท็จจริงจากโค้ดปัจจุบัน:** `isaaclab::mdp::bad_orientation()` ใน
> `deploy/include/isaaclab/envs/mdp/terminations.h` **คืนค่า `false` เสมอ** (บรรทัดที่คำนวณจริงถูก comment ไว้)
> แปลว่าใน `State_RLBase` ที่ register check "เอียงเกิน → Passive" **ไม่เคยทำงาน** — ถ้าหุ่นล้ม controller จะยังส่งคำสั่ง policy ต่อ

- [ ] แก้ `bad_orientation()` ให้ทำงานจริง (`acos(-g_z) > limit`) และทดสอบใน sim ว่าผลักหุ่นล้มแล้วเข้า Passive
- [ ] **lowstate watchdog:** ถ้าไม่ได้ `lowstate` ใหม่เกิน N ms (เช่น 50 ms) → Passive
- [ ] **Action clip:** ใส่ `clip` ใน `deploy.yaml` = joint limit ของ MJCF ลบ margin
- [ ] **NaN/Inf guard:** ถ้า output ONNX มี NaN → Passive ทันที
- [ ] **Command rate limit:** จำกัดอัตราเปลี่ยนของคำสั่งความเร็วจากจอย (เช่น 1 m/s²)
- [ ] **Joint velocity / torque guard:** ถ้า `|dq|` หรือ `|tau_est|` เกินค่า limit ต่อเนื่อง → Passive
- [ ] ทดสอบทุก guard ใน sim โดยจงใจทำให้เกิด (ฆ่า simulator, ป้อน policy ที่ output ใหญ่, ผลักล้ม)
- [ ] ยืนยันว่า `L2 + B` → Passive ทำงานจากทุก state

**เกณฑ์ผ่าน Phase 6:** ทุก guard trigger ได้ใน sim และเข้า Passive ภายใน 1 policy step

---

## Phase 7 — Sim-to-real

**ทำเมื่อ Phase 1–6 ผ่านทั้งหมดเท่านั้น** และมีคนดูแลหุ่นอย่างน้อย 2 คน

### 7.1 เตรียม

- [ ] Build `g1_ctrl` จาก commit เดียวกับที่ผ่าน sim (บันทึก git hash + md5 ของ `policy.onnx` และ `deploy.yaml`)
- [ ] ตั้ง network ของ PC: `192.168.123.222/24`, หา interface ด้วย `ip a`
- [ ] (ถ้ารันบน PC ของหุ่น Jetson/aarch64) build ใหม่บนเครื่องนั้น — CMake เลือก onnxruntime aarch64 ให้เอง
- [ ] เตรียม gantry/สายแขวน, พื้นเรียบ, พื้นที่ว่างรอบตัว ≥ 2 m

### 7.2 ขั้นตอนครั้งแรก (ทีละขั้น หยุดทันทีถ้าผิดปกติ)

1. [ ] เปิดหุ่นแบบแขวน รอเข้า **zero-torque**
2. [ ] กด `L2 + R2` → **debug mode** (ปิด locomotion service ในตัวหุ่น — `g1_ctrl` จะ error ถ้ามี process อื่นใช้ `lowcmd`)
3. [ ] รัน logger ก่อน แล้ว `./g1_ctrl --network=<iface>`
4. [ ] แขวนอยู่: `L2 + Up` → FixStand ดูว่าทุกข้อไปท่ายืนอย่างนุ่มนวล
5. [ ] แขวนอยู่: `R2 + A` → Velocity โดยไม่ขยับ stick ดูการสั่น/เสียง 30 วินาที แล้วกลับ Passive
6. [ ] เทียบ log จริงกับ log sim (S1): tracking error, torque, action smoothness — ต่างกันมากผิดปกติ → หยุดและวิเคราะห์
7. [ ] ลดหุ่นลงให้เท้าแตะพื้นโดยสายยังรับน้ำหนักบางส่วน → Velocity ยืนนิ่ง
8. [ ] ปล่อยสายหย่อน (ยังติดอยู่) → ยืนนิ่ง 60 วินาที
9. [ ] คำสั่งเล็ก vx = 0.2 → 0.4 แล้วค่อยเพิ่ม
10. [ ] ทุก session: เก็บ log, จดอุณหภูมิมอเตอร์, จดเหตุการณ์

### 7.3 Real-to-sim loop

- [ ] ใช้ logger/analysis ชุดเดียวกันเทียบ sim vs real ทีละสัญญาณ
- [ ] ถ้าต่างกันในเรื่องใด (เช่น tracking error ของข้อเท้าสูงกว่ามาก) → ปรับ MJCF ของ simulator ทดสอบให้เหมือนจริงขึ้น (system identification แบบง่าย) → ปรับ DR ตอนเทรน → วนกลับ Phase 2
- [ ] ทำให้ simulator ทดสอบเป็น "ตัวแทนหุ่นจริง" ที่แม่นขึ้นเรื่อย ๆ

---

## ตาราง Go/No-Go

| ก่อนไป | ต้องมี |
|---|---|
| Phase 2 | Policy v0 ของ Unitree เดินได้ใน sim-to-sim + logger ทำงาน |
| Phase 4 | ONNX ตรงกับ PyTorch, `deploy.yaml` สร้างจากสคริปต์ |
| Phase 5 | Parity test ผ่านทุกข้อ |
| Phase 6 | S1–S9 ผ่าน |
| Phase 7 | Safety guards ทดสอบแล้วใน sim, มี gantry และคนดูแล |
| เดินอิสระบนหุ่นจริง | ขั้น 7.2 ข้อ 1–9 ผ่าน, log จริงใกล้ sim |

---

## ภาคผนวก

### A. ไฟล์ที่ต้องสร้าง/แก้

แนะนำให้แยกโค้ดของเราไว้ใน `~/nvidia/unitree/g1_sim2sim/` และแก้ repo ต้นทางเท่าที่จำเป็น (แยก branch)

| ไฟล์ | สร้าง/แก้ | Phase |
|---|---|---|
| `g1_sim2sim/tools/dds_logger.py` | สร้าง | 1 |
| `g1_sim2sim/tools/analyze_log.py` | สร้าง | 1, 5, 7 |
| `g1_sim2sim/tools/gen_deploy_yaml.py` | สร้าง | 3 |
| `g1_sim2sim/tools/check_onnx.py` | สร้าง | 3 |
| `g1_sim2sim/tools/replay_obs.py` | สร้าง | 4 |
| `g1_sim2sim/tools/command_script.py` | สร้าง (publish คำสั่งแทนจอย) | 5 |
| `g1_sim2sim/docs/experiments.md` | สร้าง | ทุก phase |
| `unitree_rl_mjlab/src/tasks/velocity/config/g1/` (task `-Robust`) | เพิ่ม | 2 |
| `deploy/robots/g1/config/policy/velocity/v1/` | สร้าง | 3 |
| `deploy/robots/g1/config/config.yaml` (`policy_dir`) | แก้ | 3 |
| `deploy/include/FSM/State_RLBase.h` (obs dump) | แก้ | 4 |
| `deploy/include/isaaclab/envs/mdp/terminations.h` (`bad_orientation`) | แก้ | 6 |
| `simulate/src/unitree_sdk2_bridge.h` (noise, latency) | แก้ | 5 |
| `src/assets/robots/unitree_g1/xmls/scene_g1_*.xml` | สร้าง (ตัวแปร) | 5 |

### B. คำสั่งที่ใช้บ่อย

```bash
# เทรน
conda activate unitree_rl_mjlab && cd ~/nvidia/unitree/unitree_rl_mjlab
python scripts/train.py Unitree-G1-Flat --env.scene.num-envs=4096
python scripts/play.py Unitree-G1-Flat --checkpoint_file=logs/rsl_rl/g1_velocity/<run>/model_<N>.pt

# sim-to-sim (3 terminal)
cd ~/nvidia/unitree/unitree_rl_mjlab/simulate/build && ./unitree_mujoco
cd ~/nvidia/unitree/unitree_rl_mjlab/deploy/robots/g1/build && ./g1_ctrl --network=lo
python ~/nvidia/unitree/g1_sim2sim/tools/dds_logger.py --iface lo --out logs/sim_<date>.npz

# หุ่นจริง (binary เดียวกัน)
./g1_ctrl --network=<iface>
python ~/nvidia/unitree/g1_sim2sim/tools/dds_logger.py --iface <iface> --out logs/real_<date>.npz
```

### C. ปัญหาที่คาดว่าจะเจอ

| อาการ | สาเหตุที่น่าจะเป็น | ตรวจที่ |
|---|---|---|
| `g1_ctrl` ค้างที่ `Waiting for connection` | domain/interface ไม่ตรง, simulator ไม่ได้เปิด | `simulate/config.yaml` domain 0, `--network=lo` |
| `The other process is using the lowcmd channel` | มี controller/simulator อื่นส่ง lowcmd อยู่ | ปิด process เก่า |
| `Unmatched robot type` | simulator เป็นรุ่นอื่น / 23 DoF | `robot: g1`, scene 29 DoF |
| กด FSM ไม่ได้ | ไม่ได้เปิด `use_joystick` หรือ layout จอยผิด | `jstest`, `joystick_type` |
| เข้า Velocity แล้วล้มทันที | ลำดับ obs/joint ผิด, default pose ไม่ตรง | Phase 4 |
| เดินได้แต่ทิศกลับ | เครื่องหมายคำสั่ง | Phase 4 sign test |
| สั่นถี่ | kp/kd ไม่ตรง, action scale ผิด, latency | เทียบ `deploy.yaml` กับ metadata |
| ใน play ดีแต่ sim-to-sim แย่ | sim gap (frictionloss, timestep, PD) | เพิ่ม DR ใน Phase 2 |
| Controller ไม่เปลี่ยนเป็น policy ใหม่ | `parser_policy_dir` เลือก folder อื่น | ระบุ path ใน `config.yaml` |
