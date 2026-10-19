> เอกสารอ้างอิงเดิม: เก็บรายละเอียดและตัวอย่างเดิมไว้ ชื่อโครงการ `g1_training` และคำสั่งในไฟล์นี้ไม่ใช่ entry point ของโครงการใหม่ ให้เริ่มที่ [README](../../README.md) และ [แผนหลัก](../implementation_plan.md) ก่อนใช้งาน

# คู่มือทำความเข้าใจ: การเทรนหุ่นยนต์ Unitree G1 ให้ยืน/เดิน ด้วย Isaac Lab + Newton

> เอกสารนี้อธิบายโครงสร้างโฟลเดอร์/ไฟล์ และแนวคิดหลักของตัวอย่าง G1 locomotion ใน Isaac Lab
> (อ้างอิงจาก branch `develop` ของ repo `IsaacLab`)

---

## สารบัญ

1. [ภาพรวม](#1-ภาพรวม)
2. [เริ่มต้นใช้งานเร็ว (Quick Start)](#2-เริ่มต้นใช้งานเร็ว-quick-start)
3. [แนวคิดพื้นฐานที่ต้องรู้](#3-แนวคิดพื้นฐานที่ต้องรู้)
4. [โครงสร้างโฟลเดอร์และไฟล์](#4-โครงสร้างโฟลเดอร์และไฟล์)
5. [อธิบายแต่ละส่วนของ Environment (MDP)](#5-อธิบายแต่ละส่วนของ-environment-mdp)
6. [การตั้งค่า PPO (ตัวเทรน)](#6-การตั้งค่า-ppo-ตัวเทรน)
7. [การเทรนให้ "ยืน" อย่างเดียว](#7-การเทรนให้-ยืน-อย่างเดียว)
8. [ตารางสรุป: อยากแก้อะไร ต้องไปที่ไฟล์ไหน](#8-ตารางสรุป-อยากแก้อะไร-ต้องไปที่ไฟล์ไหน)
9. [ผลลัพธ์การเทรนอยู่ที่ไหน](#9-ผลลัพธ์การเทรนอยู่ที่ไหน)
10. [เอกสารอ้างอิงเพิ่มเติม](#10-เอกสารอ้างอิงเพิ่มเติม)

---

## 1. ภาพรวม

Isaac Lab มีตัวอย่างการเทรน G1 ให้เดินตามคำสั่งความเร็ว (velocity tracking) อยู่แล้ว 2 task:

| Task ID | พื้น | Physics เริ่มต้น | หมายเหตุ |
|---|---|---|---|
| `Isaac-Velocity-Flat-G1` | พื้นเรียบ (plane) | Newton (MJWarp) | เทรนเร็ว เหมาะสำหรับเริ่มต้น (1,500 iterations) |
| `Isaac-Velocity-Rough-G1` | พื้นขรุขระ (สุ่มสร้าง) | Newton (MJWarp) | ยากกว่า มี height scanner (5,000 iterations) |

**หลักการ:** หุ่นได้รับ "คำสั่งความเร็ว" แบบสุ่ม (เดินหน้า / ด้านข้าง / หมุนตัว) แล้ว policy (neural network)
ต้องเรียนรู้การขยับข้อต่อเพื่อให้ตัวหุ่นเคลื่อนที่ตามความเร็วนั้นโดยไม่ล้ม
→ เมื่อคำสั่งเป็น 0 หุ่นจะ **ยืนนิ่ง** ดังนั้น task เดียวได้ทั้ง "ยืน" และ "เดิน"

---

## 2. เริ่มต้นใช้งานเร็ว (Quick Start)

รันจาก root ของ repo (`IsaacLab/`)

```bash
# เทรนบนพื้นเรียบ (Newton เป็นค่าเริ่มต้นอยู่แล้ว)
uv run isaaclab train --task Isaac-Velocity-Flat-G1 --num_envs 4096

# เทรนบนพื้นขรุขระ
uv run isaaclab train --task Isaac-Velocity-Rough-G1

# ระบุ physics backend เองแบบชัดเจน (newton_mjwarp / physx / newton_kamino / ovphysx)
uv run isaaclab train --task Isaac-Velocity-Flat-G1 physics=newton_mjwarp

# ดูผล (play) policy ล่าสุด พร้อม visualizer ของ Newton
uv run isaaclab play --task Isaac-Velocity-Flat-G1 --num_envs 32 --checkpoint latest --viz newton
```

คำอธิบาย argument:

- `--task` : ชื่อ environment ที่ลงทะเบียนไว้ (ดูหัวข้อ 4)
- `--num_envs` : จำนวนหุ่นที่จำลองพร้อมกันบน GPU (มากขึ้น = เทรนเร็วขึ้น แต่ใช้ VRAM มากขึ้น)
- ไม่ต้องใส่ `--headless` : Isaac Lab 3.0 รันแบบไม่มีหน้าต่างเป็นค่าเริ่มต้นอยู่แล้ว ถ้าอยากดูภาพให้เพิ่ม `--viz newton` (หรือ `--viz kit`)
- `physics=...` : Hydra override สำหรับเลือก physics backend
- `--checkpoint latest` : โหลด model ล่าสุดที่เทรนไว้
- ดู option ทั้งหมด: `uv run isaaclab train --task Isaac-Velocity-Flat-G1 --help`

---

## 3. แนวคิดพื้นฐานที่ต้องรู้

### 3.1 Manager-Based Environment

Isaac Lab แบ่ง environment ออกเป็น "manager" หลายตัว แต่ละตัวดูแลเรื่องเดียว
และกำหนดค่าผ่าน **config class** (ไม่ต้องเขียน logic ใน loop เอง):

```
ManagerBasedRLEnv
 ├── Scene        → มีอะไรอยู่ในโลก (พื้น, หุ่น, เซนเซอร์, แสง)
 ├── Commands     → คำสั่งที่ให้หุ่นทำ (ความเร็วเป้าหมาย)
 ├── Actions      → policy ส่งค่าอะไรไปควบคุมหุ่น (ตำแหน่งข้อต่อ)
 ├── Observations → policy มองเห็นอะไร (ความเร็วตัว, มุมข้อต่อ, ...)
 ├── Rewards      → ให้คะแนน/หักคะแนนพฤติกรรม
 ├── Terminations → เงื่อนไขจบ episode (ล้ม, หมดเวลา)
 ├── Events       → สุ่มค่าต่าง ๆ (domain randomization) และการ reset
 └── Curriculum   → ปรับความยากตามความเก่งของ policy
```

### 3.2 การสืบทอด config (Inheritance)

config ถูกเขียนเป็นชั้น ๆ ชั้นล่างสืบทอดและ "แก้ทับ" ค่าจากชั้นบน:

```
LocomotionVelocityRoughEnvCfg   (velocity_env_cfg.py)   ← ค่ากลางสำหรับหุ่นขาทุกตัว
        │
        ▼
G1RoughEnvCfg                   (g1/rough_env_cfg.py)  ← ใส่หุ่น G1 + ปรับ reward ให้เหมาะกับหุ่นสองขา
        │
        ▼
G1FlatEnvCfg                    (g1/flat_env_cfg.py)   ← เปลี่ยนเป็นพื้นเรียบ, ตัด height scan
```

การแก้ค่าทำใน method `__post_init__()` เช่น
`self.rewards.feet_air_time.weight = 0.75`

### 3.3 Physics Preset (PhysX / Newton)

ใน `velocity_env_cfg.py` มี class `RoughPhysicsCfg(PresetCfg)` ที่รวม backend ไว้หลายแบบ:

- `newton_mjwarp` ← **ค่าเริ่มต้น** (Newton + MuJoCo-Warp solver)
- `newton_kamino`
- `physx` (Isaac Sim PhysX / OvPhysX)

เลือกตอนรันด้วย `physics=<ชื่อ>` ได้โดยไม่ต้องแก้โค้ด

### 3.4 Simulation timing

| ค่า | ความหมาย |
|---|---|
| `sim.dt = 0.005` | physics step ละ 5 ms (200 Hz) |
| `decimation = 4` | policy ตัดสินใจทุก 4 physics step → 50 Hz |
| `episode_length_s = 20.0` | 1 episode ยาว 20 วินาที |

---

## 4. โครงสร้างโฟลเดอร์และไฟล์

ไฟล์ที่เกี่ยวข้องกระจายอยู่ใน 5 package ภายใต้ `source/`:

### 4.1 Task (ส่วนที่จะแก้บ่อยที่สุด)

```
source/isaaclab_tasks/isaaclab_tasks/core/velocity/
├── velocity_env_cfg.py          # ★ config กลางของงาน locomotion (ทุกหุ่น)
├── mdp/                         # ฟังก์ชัน MDP เฉพาะงาน locomotion
│   ├── rewards.py               #   feet_air_time_positive_biped, feet_slide,
│   │                            #   stand_still_joint_deviation_l1, track_*_exp ...
│   ├── terminations.py          #   terrain_out_of_bounds
│   ├── curriculums.py           #   terrain_levels_vel (เพิ่มความยากพื้น)
│   ├── symmetry/                #   (สำหรับ symmetry augmentation)
│   └── __init__.pyi             #   export ฟังก์ชันข้างบน + ทั้งหมดจาก isaaclab.envs.mdp
└── config/
    ├── g1/                      # ★ Unitree G1
    │   ├── __init__.py          #   ลงทะเบียน task (gym.register)
    │   ├── rough_env_cfg.py     #   G1Rewards + G1RoughEnvCfg
    │   ├── flat_env_cfg.py      #   G1FlatEnvCfg
    │   └── agents/
    │       ├── rsl_rl_ppo_cfg.py        # hyperparameter PPO (RSL-RL, ค่าเริ่มต้น)
    │       ├── skrl_flat_ppo_cfg.yaml   # สำหรับ skrl
    │       └── skrl_rough_ppo_cfg.yaml
    ├── h1/  go2/  anymal_d/  cassie/    # หุ่นอื่น โครงสร้างเหมือนกัน (ใช้เป็นตัวอย่างเทียบได้)
```

**`config/g1/__init__.py`** — ผูกชื่อ task เข้ากับ config:

```python
gym.register(
    id="Isaac-Velocity-Flat-G1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:G1FlatEnvCfg",       # config ของ env
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",  # config ของตัวเทรน
        "default_agent": "rsl_rl",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
)
```

### 4.2 หุ่นยนต์ (Robot Asset)

```
source/isaaclab_assets/isaaclab_assets/robots/unitree.py
    G1_CFG          # ArticulationCfg: ไฟล์ USD, ท่าเริ่มต้น, actuator (ค่า PD gain)
    G1_MINIMAL_CFG  # G1 แบบ collision ย่อส่วน → ใช้ใน task velocity (จำลองเร็วกว่า)
    G1_29DOF_CFG    # G1 แบบ 29 DoF เต็ม (ใช้ในงาน manipulation)
```

ถ้าต้องการเปลี่ยน **ท่ายืนเริ่มต้น, stiffness/damping, effort limit** ของข้อต่อ → แก้ที่ไฟล์นี้

### 4.3 MDP กลาง (core library)

```
source/isaaclab/isaaclab/envs/
├── manager_based_rl_env.py   # ตัว environment runtime
└── mdp/
    ├── commands/             # UniformVelocityCommand (สุ่มคำสั่งความเร็ว)
    ├── actions/              # JointPositionAction
    ├── observations.py       # base_lin_vel, projected_gravity, joint_pos_rel, ...
    ├── rewards.py            # lin_vel_z_l2, action_rate_l2, joint_torques_l2, ...
    ├── terminations.py       # time_out, illegal_contact, ...
    ├── events.py             # randomize mass/friction, push, reset
    └── curriculums.py
```

### 4.4 Newton Physics Backend

```
source/isaaclab_newton/isaaclab_newton/physics/
├── mjwarp_manager_cfg.py   # ตั้งค่า solver (njmax, nconmax, integrator, ...)
└── mjwarp_manager.py
```

> `njmax` = จำนวน constraint สูงสุดต่อ env, `nconmax` = จำนวน contact สูงสุดต่อ env
> G1 Rough ตั้ง `njmax=300`, G1 Flat ตั้ง `njmax=95, nconmax=10` (พื้นเรียบมี contact น้อยกว่า → ประหยัดหน่วยความจำ)

### 4.5 Training / CLI

```
source/isaaclab/isaaclab/cli/            # คำสั่ง `uv run isaaclab train / play`
source/isaaclab_rl/isaaclab_rl/
├── entrypoints/                         # dispatch ไปยัง RL library แต่ละตัว
└── rsl_rl/
    ├── rl_cfg.py                        # base class RslRlOnPolicyRunnerCfg
    ├── vecenv_wrapper.py                # ห่อ env ให้ RSL-RL ใช้ได้
    └── exporter.py                      # export policy เป็น ONNX/JIT สำหรับ deploy
```

### 4.6 แผนภาพความสัมพันธ์

```
 uv run isaaclab train --task Isaac-Velocity-Flat-G1
            │
            ▼
   isaaclab/cli  ──►  isaaclab_rl/entrypoints  (เลือก rsl_rl)
            │
            ▼  ค้นชื่อ task ใน gym registry
   config/g1/__init__.py
      ├── env_cfg ──► flat_env_cfg.G1FlatEnvCfg
      │                  └─► rough_env_cfg.G1RoughEnvCfg
      │                        ├─► velocity_env_cfg.LocomotionVelocityRoughEnvCfg
      │                        │     ├─ mdp/ (velocity) + isaaclab.envs.mdp
      │                        │     └─ RoughPhysicsCfg ─► isaaclab_newton (MJWarp)
      │                        └─► isaaclab_assets/robots/unitree.py (G1_MINIMAL_CFG)
      └── agent_cfg ─► agents/rsl_rl_ppo_cfg.G1FlatPPORunnerCfg
            │
            ▼
   ManagerBasedRLEnv  +  RSL-RL PPO  ──►  logs/rsl_rl/g1_flat/<timestamp>/
```

---

## 5. อธิบายแต่ละส่วนของ Environment (MDP)

### 5.1 Scene — มีอะไรในโลกจำลอง

| ส่วนประกอบ | หน้าที่ |
|---|---|
| `terrain` | พื้น (Rough = สุ่มสร้างหลายระดับความยาก, Flat = plane) |
| `robot` | หุ่น G1 (`G1_MINIMAL_CFG`) |
| `height_scanner` | ยิง ray ลงพื้นรอบตัวหุ่น เพื่อ "มอง" ความสูงพื้น (เฉพาะ Rough) |
| `contact_forces` | เซนเซอร์แรงสัมผัสทุก body (ใช้ตรวจเท้าแตะพื้น / ตัวล้ม) |
| `sky_light` | แสง |

### 5.2 Commands — คำสั่งที่ให้หุ่นทำ

`UniformVelocityCommandCfg` สุ่มความเร็วเป้าหมายใหม่ทุก 10 วินาที:

| ค่า | G1 Rough | G1 Flat | ความหมาย |
|---|---|---|---|
| `lin_vel_x` | 0.0 – 1.0 | 0.0 – 1.0 | เดินหน้า (m/s) |
| `lin_vel_y` | 0.0 | -0.5 – 0.5 | เดินด้านข้าง (m/s) |
| `ang_vel_z` | -1.0 – 1.0 | -1.0 – 1.0 | หมุนตัว (rad/s) |
| `rel_standing_envs` | 0.02 | 0.02 | 2% ของ env ได้คำสั่ง "ยืนนิ่ง" |

### 5.3 Actions — policy ควบคุมอะไร

```python
joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)
```

policy ส่ง **ตำแหน่งเป้าหมายของทุกข้อต่อ** (คูณ 0.5 แล้วบวกกับท่ายืนเริ่มต้น)
จากนั้น PD controller ของ actuator จะสร้างแรงบิดให้ข้อต่อไปถึงตำแหน่งนั้น

### 5.4 Observations — policy มองเห็นอะไร

| Observation | ความหมาย |
|---|---|
| `base_lin_vel` | ความเร็วเชิงเส้นของลำตัว |
| `base_ang_vel` | ความเร็วเชิงมุมของลำตัว |
| `projected_gravity` | ทิศของแรงโน้มถ่วงเทียบกับลำตัว (บอกว่าเอียงแค่ไหน) |
| `velocity_commands` | คำสั่งความเร็วที่ได้รับ |
| `joint_pos` / `joint_vel` | มุม/ความเร็วข้อต่อ (เทียบกับท่าเริ่มต้น) |
| `actions` | action ครั้งก่อน |
| `height_scan` | ความสูงพื้นรอบตัว (Rough เท่านั้น, Flat ตัดทิ้ง) |

ทุกค่ามีการใส่ **noise** (`Unoise`) เพื่อให้ policy ทนต่อเซนเซอร์จริงที่มีสัญญาณรบกวน

### 5.5 Rewards — ให้คะแนนอะไร (หัวใจของการเทรน)

ค่า weight บวก = ให้รางวัล, ลบ = ลงโทษ (ค่าตาม `G1Rewards` + การแก้ใน `G1RoughEnvCfg`)

**รางวัลหลัก (ทำงานสำเร็จ):**

| Reward | Weight | ความหมาย |
|---|---|---|
| `track_lin_vel_xy_exp` | +1.0 | เคลื่อนที่ได้ตรงกับความเร็วที่สั่ง |
| `track_ang_vel_z_exp` | +2.0 (Flat: +1.0) | หมุนตัวได้ตรงกับที่สั่ง |
| `feet_air_time` | +0.25 (Flat: +0.75) | ยกเท้าก้าวเดินจริง (ไม่ลากเท้า) — แบบสองขา |

**การลงโทษ (ทำให้ท่าเดินสวยและปลอดภัย):**

| Reward | Weight | ความหมาย |
|---|---|---|
| `termination_penalty` | -200 | ล้ม (episode จบก่อนเวลา) |
| `feet_slide` | -0.1 | เท้าไถลขณะแตะพื้น |
| `flat_orientation_l2` | -1.0 | ลำตัวเอียง |
| `ang_vel_xy_l2` | -0.05 | ลำตัวโยก/กลิ้ง |
| `lin_vel_z_l2` | 0 (Flat: -0.2) | ลำตัวเด้งขึ้นลง |
| `action_rate_l2` | -0.005 | action เปลี่ยนกระชาก |
| `dof_acc_l2` / `dof_torques_l2` | เล็กมาก | ข้อต่อเร่ง/ใช้แรงบิดมากเกินไป |
| `dof_pos_limits` | -1.0 | ข้อเท้าชนลิมิต |
| `joint_deviation_hip/arms/fingers/torso` | -0.1 / -0.1 / -0.05 / -0.1 | ข้อต่อที่ไม่จำเป็นต่อการเดินห้ามเบี่ยงจากท่าปกติมาก |

> **เคล็ดลับ:** ถ้าหุ่นเดินท่าแปลก ๆ ให้ดูที่ reward ก่อนเสมอ ค่าแต่ละ term ถูก log ไว้ใน TensorBoard
> (`Episode_Reward/<ชื่อ term>`) ทำให้เห็นว่า term ไหนครอบงำการเรียนรู้

### 5.6 Terminations — จบ episode เมื่อไร

| Term | ความหมาย |
|---|---|
| `time_out` | ครบ 20 วินาที (ไม่ถือว่าล้มเหลว) |
| `base_contact` | ลำตัว (`torso_link`) แตะพื้น = ล้ม |

### 5.7 Events — Domain Randomization

ทำให้ policy ทนทานพอจะนำไปใช้กับหุ่นจริง (sim-to-real):

| Event | เวลา | ทำอะไร |
|---|---|---|
| `physics_material` | startup | กำหนดค่าแรงเสียดทาน |
| `add_base_mass` | startup | สุ่มมวลลำตัว ±25% |
| `reset_base` | reset | สุ่มตำแหน่ง/ทิศ/ความเร็วเริ่มต้น |
| `reset_robot_joints` | reset | ท่าข้อต่อเริ่มต้น (G1 ใช้ท่า default ตรง ๆ) |
| `push_robot` | ทุก 10–15 วิ | ผลักหุ่นแบบสุ่ม ฝึกการทรงตัว |

### 5.8 Curriculum

`terrain_levels_vel` (Rough เท่านั้น): ถ้าหุ่นเดินได้ไกล → ย้ายไปพื้นที่ยากขึ้น, ถ้าเดินไม่ได้ → ย้ายไปพื้นง่ายลง

---

## 6. การตั้งค่า PPO (ตัวเทรน)

ไฟล์: `config/g1/agents/rsl_rl_ppo_cfg.py`

| ค่า | Rough | Flat | ความหมาย |
|---|---|---|---|
| `max_iterations` | 5000 | 1500 | จำนวนรอบการเทรน |
| `num_steps_per_env` | 24 | 24 | จำนวน step ที่เก็บข้อมูลต่อ env ต่อรอบ |
| `actor/critic hidden_dims` | [512, 256, 128] | [256, 128, 128] | ขนาด neural network |
| `learning_rate` | 1e-3 (adaptive) | เหมือนกัน | ปรับอัตโนมัติตาม `desired_kl=0.01` |
| `gamma` / `lam` | 0.99 / 0.95 | เหมือนกัน | discount factor / GAE |
| `save_interval` | 50 | 50 | บันทึก checkpoint ทุก 50 รอบ |
| `experiment_name` | `g1_rough` | `g1_flat` | ชื่อโฟลเดอร์ log |

---

## 7. การเทรนให้ "ยืน" อย่างเดียว

ไม่มี task "ยืน" แยกต่างหาก แต่ทำได้ 2 วิธี:

### วิธีที่ 1: Override ผ่าน command line (ไม่ต้องแก้โค้ด)

```bash
uv run isaaclab train --task Isaac-Velocity-Flat-G1 \
  env.commands.base_velocity.ranges.lin_vel_x="[0.0,0.0]" \
  env.commands.base_velocity.ranges.lin_vel_y="[0.0,0.0]" \
  env.commands.base_velocity.ranges.ang_vel_z="[0.0,0.0]" \
  env.rewards.feet_air_time.weight=0.0
```

> ปิด `feet_air_time` เพราะ reward นี้กระตุ้นให้ยกเท้า ซึ่งขัดกับการยืนนิ่ง

### วิธีที่ 2: สร้าง task ใหม่ (แนะนำถ้าจะใช้ซ้ำ)

1. สร้างไฟล์ `config/g1/stand_env_cfg.py`:

   ```python
   from isaaclab.utils import configclass

   from .flat_env_cfg import G1FlatEnvCfg


   @configclass
   class G1StandEnvCfg(G1FlatEnvCfg):
       """G1 standing-only environment."""

       def __post_init__(self):
           super().__post_init__()
           # คำสั่งความเร็วเป็นศูนย์เสมอ
           self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
           self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
           self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
           self.commands.base_velocity.rel_standing_envs = 1.0
           # ไม่ต้องให้รางวัลการยกเท้า
           self.rewards.feet_air_time = None
   ```

2. ลงทะเบียนใน `config/g1/__init__.py`:

   ```python
   gym.register(
       id="Isaac-Stand-Flat-G1",
       entry_point="isaaclab.envs:ManagerBasedRLEnv",
       disable_env_checker=True,
       kwargs={
           "env_cfg_entry_point": f"{__name__}.stand_env_cfg:G1StandEnvCfg",
           "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
           "default_agent": "rsl_rl",
       },
   )
   ```

3. รัน: `uv run isaaclab train --task Isaac-Stand-Flat-G1`

> อาจเพิ่ม reward `stand_still_joint_deviation_l1` (มีอยู่แล้วใน `core/velocity/mdp/rewards.py`)
> เพื่อลงโทษการขยับข้อต่อเมื่อคำสั่งเป็นศูนย์

---

## 8. ตารางสรุป: อยากแก้อะไร ต้องไปที่ไฟล์ไหน

| ต้องการ | ไฟล์ |
|---|---|
| แก้ reward / penalty ของ G1 | `core/velocity/config/g1/rough_env_cfg.py` (`G1Rewards`) |
| แก้ค่าเฉพาะพื้นเรียบ / Newton solver | `core/velocity/config/g1/flat_env_cfg.py` |
| แก้ช่วงคำสั่งความเร็ว, observation, randomization (ทุกหุ่น) | `core/velocity/velocity_env_cfg.py` |
| เพิ่มฟังก์ชัน reward ใหม่ | `core/velocity/mdp/rewards.py` |
| แก้ hyperparameter PPO | `core/velocity/config/g1/agents/rsl_rl_ppo_cfg.py` |
| แก้ PD gain / ท่าเริ่มต้นของหุ่น | `isaaclab_assets/robots/unitree.py` |
| เพิ่ม task ใหม่ | `core/velocity/config/g1/__init__.py` (`gym.register`) |
| ตั้งค่า physics backend | `velocity_env_cfg.py` (`RoughPhysicsCfg`) หรือใช้ `physics=...` ตอนรัน |

*(path ใน `core/velocity/...` อยู่ภายใต้ `source/isaaclab_tasks/isaaclab_tasks/`)*

---

## 9. ผลลัพธ์การเทรนอยู่ที่ไหน

```
logs/rsl_rl/<experiment_name>/<timestamp>/
├── model_<iteration>.pt     # checkpoint
├── params/                  # config ที่ใช้เทรน (env.yaml, agent.yaml)
├── events.out.tfevents.*    # log สำหรับ TensorBoard
└── exported/                # policy.onnx / policy.pt (สร้างตอนรัน play)
```

ดูกราฟการเทรน:

```bash
uv run tensorboard --logdir logs/rsl_rl/g1_flat
```

ค่าที่ควรดู:

- `Train/mean_reward` — ควรเพิ่มขึ้นเรื่อย ๆ
- `Train/mean_episode_length` — ใกล้ 1000 step (= 20 วิ) แปลว่าหุ่นไม่ล้ม
- `Episode_Reward/track_lin_vel_xy_exp` — ความแม่นในการเดินตามคำสั่ง
- `Episode_Termination/base_contact` — ควรลดลง (ล้มน้อยลง)

---

## 10. เอกสารอ้างอิงเพิ่มเติม

| เอกสาร | เนื้อหา |
|---|---|
| `docs/source/setup/quickstart.rst` | พื้นฐาน train / play |
| `docs/source/features/hydra.rst` | วิธี override config จาก command line |
| `docs/source/how-to/transfer_policies_between_physx_and_newton.rst` | ย้าย policy ระหว่าง PhysX ↔ Newton (มีตัวอย่าง G1 Rough) |
| `skills/` ใน repo | คู่มือเชิงลึก เช่น `isaaclab-training-rl-agents`, `isaaclab-debugging-rl-training`, `isaaclab-selecting-backends` |
| `core/velocity/config/h1/` | ตัวอย่างหุ่น humanoid อีกตัว ใช้เปรียบเทียบได้ |
