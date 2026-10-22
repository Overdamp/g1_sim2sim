# แผนการทำขั้นที่ 2: การเพิ่ม Domain Randomization เชิงลึกและ Fine-Tune Policy สำหรับหุ่นจริง Unitree G1 (23 DoF)

> **สถานะปัจจุบัน:** ผ่านขั้นที่ 1 แล้ว — โมเดล `Unitree-G1-23Dof-Flat` (เทรนครบ 10,000 iterations ในโฟลเดอร์ `2026-10-20_18-26-38`) ทดสอบยืนและเดินใน `unitree_mujoco` คู่กับ C++ Controller (`g1_ctrl`) ได้อย่างนิ่งและเสถียรแล้ว
>
> **เป้าหมายของเอกสารนี้:** อธิบายแผนการปรับปรุงสภาพแวดล้อมการเทรนใน **ขั้นที่ 2 (Phase 2 Sim-to-Real Hardening)** ก่อนลงมือแก้ไขโค้ดจริง เพื่อให้ Policy ทนทานต่อความแตกต่างทางฟิสิกส์ระหว่างโมเดลใน Simulator กับหุ่นยนต์ Unitree G1 23 DoF ตัวจริง

---

## 1. ทำไมต้องมีขั้นที่ 2? (ช่องว่างระหว่าง Sim กับหุ่นจริงที่ตรวจพบ)

จากการตรวจสอบเทียบกันระหว่าง **คู่มือฮาร์ดแวร์ Unitree G1 (PDF 7 ฉบับ)**, **ไฟล์ที่ใช้เทรน (`g1_23dof.xml`)**, และ **ไฟล์จำลองหุ่นจริง (`scene_g1_23dof.xml`)** พบว่าแม้โครงสร้าง 23 ข้อต่อ, ลิมิตมุม, แรงบิด และความเร็วจะตรงกันแล้ว แต่ยังมี **ความต่างทางฟิสิกส์ 3 จุดสำคัญ** ที่โมเดลขั้นที่ 1 ยังไม่เคยเจอระหว่างเทรน:

| จุดที่ต่างกัน | ค่าในไฟล์ที่ใช้เทรน (`g1_23dof.xml`) | ค่าใน `scene_g1_23dof.xml` และหุ่นจริง G1 23 DoF | ผลกระทบหากไม่เทรนเพิ่ม |
| :--- | :--- | :--- | :--- |
| **1. น้ำหนักลำตัวและแบตเตอรี่ (`torso_link` mass)** | ลำตัวหนัก `7.818 kg`<br>(รวมทั้งตัว **`32.11 kg`**) | เมื่อใส่แบตเตอรี่จริง (`421.2Wh`) ลำตัวหนัก `~9.84 kg`<br>(รวมทั้งตัว **`~34.1–35.0 kg`** ตามคู่มือหน้า 14) | หุ่นจริงหนักกว่าใน Sim ประมาณ **2–3 kg** อาจทำให้เข่าย่อต่ำกว่าปกติหรือโน้มตัวเมื่อหยุดเดิน |
| **2. ความฝืดและแรงเสียดทานในชุดเกียร์ (`damping` & `frictionloss`)** | ข้อต่อลื่น 100%<br>(`damping = 0.0`, `frictionloss = 0.0`) | มีความฝืดของเกียร์ Planetary และก้านกระทุ้งคู่ที่ข้อเท้า<br>(`damping = 0.05`, `frictionloss = 0.2`) | ตอนยกเท้าหรือเปลี่ยนทิศทางช้าๆ มอเตอร์จริงอาจติดแรงเสียดทานสถิต (Stiction) ทำให้ก้าวสั้นกว่าที่ Policy คาดไว้ |
| **3. การแกว่งของกำลังมอเตอร์ (`kp`, `kd` และ `effort_limit`)** | ค่า `kp`, `kd` และแรงบิดสูงสุดคงที่เป๊ะ 100% ตลอดเวลา | แรงดันแบตเตอรี่ลดลงจาก `54.6V` $\rightarrow$ `46.8V` $\rightarrow$ `39V` และอุณหภูมิมอเตอร์สูงขึ้นขณะเดิน | การตอบสนองของมอเตอร์จริงอาจอ่อนลงหรือแข็งขึ้นราว **$\pm 10\text{–}15\%$** จากค่าทฤษฎี |

---

## 2. สรุป Domain Randomization ทั้งหมด (เดิม + ที่จะเพิ่มใหม่ในขั้นที่ 2)

การเพิ่ม Domain Randomization ในขั้นที่ 2 **จะไม่เปลี่ยนโครงสร้างเครือข่าย Neural Network** (Actor ยังคงรับ Input `80` มิติ และส่งออก `23` มิติเหมือนเดิมทุกประการ) จึงสามารถ **Resume (เทรนต่อยอด)** จากโมเดล `model_10000.pt` เดิมได้ทันที

### 2.1 รายการที่มีอยู่แล้วตั้งแต่ขั้นที่ 1 (คงไว้เหมือนเดิม)
1. **`foot_friction`**: สุ่มความฝืดพื้นรองเท้า 14 จุด ในช่วง `0.3 ถึง 1.6`
2. **`base_com`**: สุ่มเลื่อนจุดศูนย์ถ่วง (CoM) ของ `torso_link` ในแกน X, Y, Z แกนละ `-0.05 ถึง +0.05 m` ($\pm 5\text{ cm}$)
3. **`encoder_bias`**: สุ่มค่าเพี้ยนศูนย์ของ Encoder ทุกข้อต่อ ในช่วง `-0.015 ถึง +0.015 rad` ($\approx \pm 0.86^\circ$)
4. **`push_robot`**: สุ่มผลัก/กระชากลำตัวหุ่นทุกๆ `5–6 วินาที` (`vx, vy ±0.5 m/s`, `vz ±0.4 m/s`)
5. **Actor Observation Noise**: สุ่มสัญญาณรบกวนทุกๆ step ที่ Gyro (`±0.2 rad/s`), Gravity (`±0.05`), มุมข้อต่อ (`±0.01 rad`), และความเร็วข้อต่อ (`±1.5 rad/s`)

### 2.2 รายการที่จะเพิ่มใหม่ในขั้นที่ 2 (5 Events ใหม่)

| ลำดับ | ชื่อ Event ที่จะเพิ่ม | ฟังก์ชันใน `mjlab.envs.mdp.dr` | ชิ้นส่วนเป้าหมาย | ค่าพารามิเตอร์ที่จะตั้ง | เหตุผลทางวิศวกรรม |
| :---: | :--- | :--- | :--- | :--- | :--- |
| **1** | **`torso_mass_payload`** | `dr.body_mass` | `torso_link` | `operation="add"`<br>`ranges=(-1.0, 3.5)` *(kg)* | ให้หุ่นเจอตั้งแต่น้ำหนักลำตัวเบาลง `1 kg` ไปจนถึงหนักขึ้น `+3.5 kg` (ครอบคลุมน้ำหนักรวม `31.1 – 35.6 kg` ซึ่งคลุมน้ำหนักแบตเตอรี่จริง `~35 kg` ได้สบาย) |
| **2** | **`limb_mass_scale`** | `dr.body_mass` | ทุกชิ้นส่วนยกเว้น `torso_link` (`^(?!torso_link$).*`) | `operation="scale"`<br>`ranges=(0.90, 1.10)` | สุ่มมวลของสะโพก ขา เท้า และแขน/มือยาง ในช่วง **$\pm 10\%$** เพื่อชดเชยความคลาดเคลื่อนของน้ำหนักชิ้นส่วนจริงและสายไฟ |
| **3** | **`joint_friction`** | `dr.joint_friction` | ข้อต่อทั้งหมด (`.*`) | `operation="abs"`<br>`ranges=(0.0, 0.30)` | สุ่มค่า `dof_frictionloss` ตั้งแต่ลื่นสนิท (`0.0`) ไปจนถึงฝืดกว่าปกติ (`0.30`) ซึ่งคลุมค่า `0.2` ใน `scene_g1_23dof.xml` และความฝืดเกียร์จริง |
| **4** | **`joint_damping`** | `dr.joint_damping` | ข้อต่อทั้งหมด (`.*`) | `operation="abs"`<br>`ranges=(0.0, 0.10)` | สุ่มค่าความหนืดเชิงกล `dof_damping` ตั้งแต่ `0.0 ถึง 0.10` (คลุมค่า `0.05` ใน `scene_g1_23dof.xml`) |
| **5** | **`motor_pd_gains`** & **`motor_effort_limits`** | `dr.pd_gains`<br>`dr.effort_limits` | มอเตอร์ทั้ง 23 ตัว (`robot`) | `kp_range=(0.85, 1.15)`<br>`kd_range=(0.85, 1.15)`<br>`effort_limit_range=(0.85, 1.05)` | สุ่มความแข็ง/ความหนืดของ PD Controller ในช่วง **$\pm 15\%$** และเพดานแรงบิดสูงสุดในช่วง **`85% – 105%`** เพื่อจำลองภาวะแบตเตอรี่อ่อนหรือมอเตอร์ร้อน |

---

## 3. ไฟล์ที่จะแก้ไขและรายละเอียดโค้ด

เราจะแก้ไขเพียง **1 ไฟล์** คือ:
* [`/home/luke/nvidia/unitree/unitree_rl_mjlab/src/tasks/velocity/config/g1_23dof/env_cfgs.py`](file:///home/luke/nvidia/unitree/unitree_rl_mjlab/src/tasks/velocity/config/g1_23dof/env_cfgs.py)

### โค้ดที่จะเพิ่มในฟังก์ชัน `unitree_g1_23dof_rough_env_cfg()` (ซึ่ง `unitree_g1_23dof_flat_env_cfg` สืบทอดไปใช้โดยตรง):

```python
from mjlab.envs.mdp import dr
from mjlab.managers.scene_entity_config import SceneEntityCfg

# --- Phase 2: Sim-to-Real Domain Randomization ---
# 1. สุ่มเพิ่ม/ลดมวลลำตัว (ชดเชยน้ำหนักแบตเตอรี่จริง ~2-3 kg)
cfg.events["torso_mass_payload"] = EventTermCfg(
  mode="startup",
  func=dr.body_mass,
  params={
    "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
    "operation": "add",
    "ranges": (-1.0, 3.5),
  },
)

# 2. สุ่มสเกลมวลของแขน ขา และสะโพก (±10%)
cfg.events["limb_mass_scale"] = EventTermCfg(
  mode="startup",
  func=dr.body_mass,
  params={
    "asset_cfg": SceneEntityCfg("robot", body_names=(r"^(?!torso_link$).*",)),
    "operation": "scale",
    "ranges": (0.90, 1.10),
  },
)

# 3. สุ่มแรงเสียดทานสถิตในชุดเกียร์ข้อต่อ (ครอบคลุมค่า 0.2 ใน scene_g1_23dof.xml)
cfg.events["joint_friction"] = EventTermCfg(
  mode="startup",
  func=dr.joint_friction,
  params={
    "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
    "operation": "abs",
    "ranges": (0.0, 0.30),
  },
)

# 4. สุ่มความหนืดเชิงกลของข้อต่อ (ครอบคลุมค่า 0.05 ใน scene_g1_23dof.xml)
cfg.events["joint_damping"] = EventTermCfg(
  mode="startup",
  func=dr.joint_damping,
  params={
    "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
    "operation": "abs",
    "ranges": (0.0, 0.10),
  },
)

# 5. สุ่มค่า Kp / Kd ของมอเตอร์ (±15%) และเพดานแรงบิด (85% - 105%)
cfg.events["motor_pd_gains"] = EventTermCfg(
  mode="startup",
  func=dr.pd_gains,
  params={
    "asset_cfg": SceneEntityCfg("robot"),
    "operation": "scale",
    "kp_range": (0.85, 1.15),
    "kd_range": (0.85, 1.15),
  },
)
cfg.events["motor_effort_limits"] = EventTermCfg(
  mode="startup",
  func=dr.effort_limits,
  params={
    "asset_cfg": SceneEntityCfg("robot"),
    "operation": "scale",
    "effort_limit_range": (0.85, 1.05),
  },
)
```

---

## 4. ขั้นตอนการลงมือทำและทดสอบเปรียบเทียบ (Step-by-Step Workflow)

เมื่อคุณอนุมัติแผนนี้ เราจะดำเนินการตามลำดับที่ปลอดภัยและตรวจสอบย้อนกลับได้ทุกขั้นตอนดังนี้ครับ:

### ขั้นตอนที่ 1: สำรองโมเดลขั้นที่ 1 ไว้ก่อน (Backup Phase 1 Policy)
ก่อนเทรนเพิ่ม เราจะสำรองไฟล์ `policy.onnx` ตัวปัจจุบันที่เทสแล้วว่านิ่งไว้เป็น `policy_phase1_10000.onnx`:
```bash
cp /home/luke/nvidia/unitree/unitree_rl_mjlab/deploy/robots/g1_23dof/config/policy/velocity/v0/exported/policy.onnx \
   /home/luke/nvidia/unitree/unitree_rl_mjlab/deploy/robots/g1_23dof/config/policy/velocity/v0/exported/policy_phase1_10000.onnx
```
*(หากเมื่อไหร่ต้องการสลับกลับไปใช้ตัวเดิม ก็สามารถก๊อปปี้กลับมาทับได้ทันที)*

### ขั้นตอนที่ 2: แก้ไขไฟล์ `env_cfgs.py` และรัน Smoke Test (1 Iteration)
* เพิ่ม Domain Randomization ทั้ง 5 รายการลงใน `src/tasks/velocity/config/g1_23dof/env_cfgs.py`
* รันทดสอบ 1 iteration เพื่อยืนยันว่าไม่มี Error และ `mujoco-warp` อัปเดตฟิลด์ของโมเดล (`body_mass`, `dof_frictionloss`, `dof_damping`, `actuator_gainprm`, `actuator_forcerange`) บน GPU ได้ถูกต้อง

### ขั้นตอนที่ 3: รัน Fine-Tuning ต่อจาก `model_10000.pt` อีก 3,000 Iterations (Iteration 10,000 $\rightarrow$ 13,000)
ใช้คำสั่ง `--agent.resume=True` เพื่อโหลดทั้ง **Actor และ Critic** จาก `2026-10-20_18-26-38` (`model_10000.pt`) มาฝึกปรับตัวต่ออีก 3,000 iterations (ใช้เวลาประมาณ **~40–45 นาที** บนเครื่องของคุณ หรือประมาณ **~5–10 นาที** บน DGX):
```bash
cd /home/luke/nvidia/unitree/unitree_rl_mjlab
python scripts/train.py Unitree-G1-23Dof-Flat \
  --env.scene.num-envs=1024 \
  --agent.resume=True \
  --agent.load-run=2026-10-20_18-26-38 \
  --agent.load-checkpoint=model_10000.pt \
  --agent.max-iterations=3000 \
  --agent.logger=tensorboard
```
*(หมายเหตุ: ใน `rsl_rl` ค่า `--agent.max-iterations=3000` เมื่อใช้คู่กับ `--agent.resume=True` จะหมายถึงการเทรนบวกเพิ่มไปอีก 3,000 iterations จาก 10,000 ไปจบที่ 13,000 ครับ)*

### ขั้นตอนที่ 4: นำ `policy.onnx` ใหม่มาทดสอบใน `unitree_mujoco` + `g1_ctrl`
เมื่อเทรนครบ 13,000 iterations:
1. ก๊อปปี้ `policy.onnx` ตัวใหม่ไปวางที่ `deploy/robots/g1_23dof/config/policy/velocity/v0/exported/policy.onnx`
2. ทดสอบเดินใน `unitree_mujoco` คู่กับ `g1_ctrl` เพื่อเปรียบเทียบความนิ่งและดูการตอบสนองเมื่อเพิ่มน้ำหนักลำตัวจำลองแบตเตอรี่จริง
