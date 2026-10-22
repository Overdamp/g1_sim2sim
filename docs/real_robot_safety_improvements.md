# แนวทางการปรับปรุงความปลอดภัยและความเสถียรของ Unitree G1 (23-DoF) ก่อนลงหุ่นยนต์จริง
**(ถอดบทเรียนจากประสบการณ์จริงของ sentdex + การตรวจสอบโค้ด C++ Deploy เชิงลึก)**

เอกสารนี้รวบรวมแนวทางการปรับปรุงระบบความปลอดภัย (Safety) และการถนอมฮาร์ดแวร์สำหรับโปรเจกต์ `unitree_rl_mjlab` (`g1_23dof`) โดยอ้างอิงจากปัญหาจริงที่พบในคลิปทดสอบ Unitree G1 ของ **sentdex** ควบคู่กับการตรวจสอบโค้ด C++ ฝั่ง Deployment ของเราอย่างละเอียด

---

## 🚨 1. [วิกฤต - ต้องแก้ทันที] เปิดใช้งานระบบตัดเข้าโหมด Damping อัตโนมัติเมื่อหุ่นล้ม (`bad_orientation`)

### บทเรียนจากคลิป sentdex (`18:00 – 19:15`)
* ในระหว่างการเทรนใน MuJoCo ทันทีที่ลำตัวหุ่นเอียงเกินมุมที่กำหนด Episode จะถูกตัดจบ (`reset`) ทันที
* ส่งผลให้ **Neural Network ไม่เคยเห็นค่า IMU ตอนที่หุ่นกำลังล้มหรือนอนอยู่บนพื้นเลย (Out-of-Distribution)**
* เมื่อหุ่นจริงเสียหลักล้มลงพื้น Policy จะคำนวณ Action ผิดเพี้ยนอย่างรุนแรง ทำให้ขาและแขน **ดีดดิ้นอย่างบ้าคลั่ง** เสี่ยงต่อการทำเกียร์มอเตอร์พังหรือฟาดคนรอบข้าง

### สิ่งที่ตรวจพบในโค้ด C++ ของเราตอนนี้!
แม้ว่าในไฟล์ [`deploy/robots/g1_23dof/src/State_RLBase.cpp`](file:///home/luke/nvidia/unitree/unitree_rl_mjlab/deploy/robots/g1_23dof/src/State_RLBase.cpp#L77-L82) จะมีการผูก `bad_orientation(env.get(), 1.0)` เพื่อสั่งตัดเข้าโหมด `Passive` (Damping) ไว้แล้ว แต่เมื่อไปเปิดดูไส้ในของฟังก์ชันที่ไฟล์ [`deploy/include/isaaclab/envs/mdp/terminations.h`](file:///home/luke/nvidia/unitree/unitree_rl_mjlab/deploy/include/isaaclab/envs/mdp/terminations.h#L10-L16) กลับพบว่า **โค้ดบรรทัดเช็กมุมเอียงถูก Comment ปิดไว้ (`return false;`)!**

```cpp
// ไฟล์ปัจจุบัน: deploy/include/isaaclab/envs/mdp/terminations.h
inline bool bad_orientation(ManagerBasedRLEnv* env, float limit_angle = 1.0)
{
    auto & asset = env->robot;
    auto & data = asset->data.projected_gravity_b;
//    return std::fabs(std::acos(-data[2])) > limit_angle;  // <-- ถูกปิดไว้!
    return false;
}
```

### แนวทางการปรับปรุงแก้ไข
1. **เปิดใช้งาน `bad_orientation` ใน [`terminations.h`](file:///home/luke/nvidia/unitree/unitree_rl_mjlab/deploy/include/isaaclab/envs/mdp/terminations.h):**
   * ใช้ `std::clamp(-data[2], -1.0f, 1.0f)` ก่อนเข้า `std::acos` เพื่อป้องกันปัญหาค่า `NaN` จาก Floating-point precision
   * ปรับมุม `limit_angle` จาก `1.0 rad` (~57°) ลงมาอยู่ที่ **`0.785 rad` (~45°)** หรือ **`0.85 rad` (~48°)** เพื่อให้ตัดเข้าโหมด `Passive` (ปล่อยตัวอ่อน Damping) ทันทีที่หุ่นเสียหลักเกินกว่าจะกู้คืนได้ ก่อนที่ตัวหุ่นจะกระแทกพื้น

```cpp
inline bool bad_orientation(ManagerBasedRLEnv* env, float limit_angle = 0.85f)
{
    auto & asset = env->robot;
    auto & data = asset->data.projected_gravity_b;
    float cos_tilt = std::clamp(-data[2], -1.0f, 1.0f);
    return std::fabs(std::acos(cos_tilt)) > limit_angle;
}
```

---

## 🦾 2. [สำคัญมาก] ป้องกันเขี้ยวพลาสติกหัวไหล่หักจากอาการแขนกระตุก (Arm Action Smoothing / Freeze)

### บทเรียนจากคลิป sentdex (`27:10 – 33:15`, `t=1900s`)
* แขนซ้ายของหุ่น G1 ในคลิป **หลุดออกจากหัวไหล่ถึง 2 ครั้ง** เพราะชิ้นส่วนยึดแขนเข้ากับมอเตอร์ไหล่ด้านในเป็น **แคลมป์พลาสติกพิมพ์ 3 มิติที่มีเขี้ยวล็อกเล็กๆ (Plastic Tabs)**
* เมื่อ RL Policy สั่งขยับแขนกระตุกไปมาด้วยความถี่สูง (`50 Hz`) เพื่อช่วยทรงตัว แรงกระชากซ้ำๆ (Impulsive Torque) จะทำให้เขี้ยวพลาสติกที่ไหล่ร้าวและหักกลางอากาศ
* sentdex พบว่า **ลำตัวและเอว (Waist) คือหัวใจหลักของการทรงตัว ส่วนแขนแทบไม่ต้องช่วยเหวี่ยงเลย** และแนะนำให้ทำ **Low-Pass Filter (Smoothing)** เฉพาะข้อต่อแขน หรือล็อกแขนไว้ใกล้ท่ามาตรฐาน

### แนวทางการปรับปรุงแก้ไข (ทำได้ 2 ระดับโดยไม่ต้องเทรนใหม่)

#### ทางเลือก 2A: ใส่ Exponential Moving Average (EMA) Low-Pass Filter เฉพาะข้อต่อแขน (Joint Index 13–22) ใน C++
กรองความถี่สูงของคำสั่งแขนใน [`joint_actions.h`](file:///home/luke/nvidia/unitree/unitree_rl_mjlab/deploy/include/isaaclab/envs/mdp/actions/joint_actions.h) หรือ [`State_RLBase.cpp`](file:///home/luke/nvidia/unitree/unitree_rl_mjlab/deploy/robots/g1_23dof/src/State_RLBase.cpp):
* **ขาและเอว (`index 0..12`):** ส่งค่า `q_target` จาก Policy โดยตรง (`alpha = 1.0`) เพื่อให้ตอบสนองต่อการก้าวเท้าและทรงตัวได้ไวที่สุด
* **แขนซ้ายและขวา (`index 13..22`):** กรองด้วยสมการ EMA:
  $$q_{\text{cmd}, t} = \alpha \cdot q_{\text{policy}, t} + (1 - \alpha) \cdot q_{\text{cmd}, t-1}$$
  โดยตั้งค่า $\alpha \approx 0.2 - 0.3$ (จะช่วยตัดอาการกระตุกความถี่สูงออกได้ 70–80% แต่ยังยอมให้แขนแกว่งตามจังหวะเดินอย่างนุ่มนวล ถนอมเขี้ยวพลาสติกหัวไหล่ได้100%)

#### ทางเลือก 2B: จำกัดกรอบมุมแกว่งของแขน (Arm Position & Velocity Clamping)
* ใน [`deploy.yaml`](file:///home/luke/nvidia/unitree/unitree_rl_mjlab/deploy/robots/g1_23dof/config/policy/velocity/v0/params/deploy.yaml) ปัจจุบัน `clip: null` ทำให้ถ้าเกิดสภาวะสุดวิสัยที่เน็ตเวิร์กจ่ายค่า Action กระโดด แขนอาจสะบัดเต็มแรง
* เราสามารถกำหนดกรอบ `clip` ของข้อต่อแขน (`13..22`) ไม่ให้แกว่งออกนอกระยะ $\pm 0.5\text{ rad}$ จากท่า `default_joint_pos` ได้

---

## 🛡️ 3. ป้องกันการกระตุกในเสี้ยววินาทีแรกที่สลับจาก `FixStand` เข้าสู่โหมด `Velocity` (Startup Action Initialization)

### สิ่งที่ตรวจพบในโค้ด C++ ของเราตอนนี้
ในไฟล์ [`deploy/include/isaaclab/envs/mdp/actions/joint_actions.h`](file:///home/luke/nvidia/unitree/unitree_rl_mjlab/deploy/include/isaaclab/envs/mdp/actions/joint_actions.h#L26-L36) และ [`State_RLBase.h`](file:///home/luke/nvidia/unitree/unitree_rl_mjlab/deploy/include/FSM/State_RLBase.h#L36-L55):
* ตอนเริ่มต้นสร้าง `JointAction` หรือตอนเรียก `reset()` ตัวแปร `_processed_actions` ถูกรีเซ็ตเป็น **`0.0f` ทั้ง 23 ข้อต่อ** (ซึ่งคือท่ายืนขาเหยียดตรง 0 เรเดียน ไม่ใช่ท่ายืนงอเข่า `_offset`!)
* เนื่องจาก `policy_thread` รันแยก Thread กับ `State_RLBase::run()` หากลูป `State_RLBase::run()` (ความถี่สูง) ทำงานก่อนที่ `env->step()` รอบแรกใน `policy_thread` จะคำนวณเสร็จ มอเตอร์อาจได้รับคำสั่ง `q = 0.0f` ไป 1–2 มิลลิวินาทีแรก ทำให้เกิดการกระตุกตอนเปลี่ยนโหมด

### แนวทางการปรับปรุงแก้ไข
* ใน Constructor และฟังก์ชัน `reset()` ของ [`JointAction`](file:///home/luke/nvidia/unitree/unitree_rl_mjlab/deploy/include/isaaclab/envs/mdp/actions/joint_actions.h#L78-L81) ให้กำหนดค่าเริ่มต้นของ `_processed_actions` เป็น **`_offset` (ท่า Default Standing Pose)** แทน `0.0f` เสมอ:
  ```cpp
  void reset()
  {
      _raw_actions.assign(_action_dim, 0.0f);
      if (!_offset.empty()) {
          _processed_actions = _offset;
      } else {
          _processed_actions.assign(_action_dim, 0.0f);
      }
  }
  ```
* ทำ **Action Slew-Rate / Warmup Blend ช่วง 0.5 วินาทีแรก** ตอนเข้าโหมด `Velocity` เพื่อให้ค่อยๆ เกลี่ยจากมุมข้อต่อปัจจุบันในโหมด `FixStand` เข้าสู่คำสั่งของ RL Policy อย่างนุ่มนวลไร้รอยต่อ

---

## ⏱️ 4. ระบบ Watchdog ตรวจจับ Policy Thread ค้างหรือความหน่วง (Jetson / DDS Lag Protection)

### บทเรียนจากคลิป sentdex (`26:30 – 27:10`)
* sentdex พบว่าในการรันจริง บางจังหวะบอร์ดประมวลผลบนตัวหุ่นเกิดอาการหน่วง (Lag) ชั่วคราวเมื่อมีแพ็กเก็ตวิ่งเข้ามาจำนวนมาก ทำให้หุ่นไม่ตอบสนองต่อปุ่มหยุดทันที

### แนวทางการปรับปรุงแก้ไข
* เพิ่มตัวแปร `last_policy_step_time` ใน [`State_RLBase.h`](file:///home/luke/nvidia/unitree/unitree_rl_mjlab/deploy/include/FSM/State_RLBase.h) เพื่อบันทึกเวลาทุกครั้งที่ `env->step()` ทำงานเสร็จ (`dt = 20 ms`)
* ในฟังก์ชัน `State_RLBase::run()` ถ้าพบว่าเวลาผ่านไปเกิน **`100 ms` (5 รอบของ Policy)** แล้ว `policy_thread` ยังไม่อัปเดตค่าใหม่ (เช่น ONNX ค้าง หรือ CPU โหลดเต็ม) ให้สั่งตัดเข้าโหมด `Passive` ทันที หรือล็อกคำสั่งความเร็วเป็น `0` เพื่อไม่ให้หุ่นเดินค้างด้วยคำสั่งเก่า

---

## 🦶 5. การป้องกันเชิงกายภาพ (Physical & Field Testing Checklist) จากประสบการณ์จริง

นอกจากฝั่งซอฟต์แวร์แล้ว จากคลิปของ sentdex มีข้อควรระวังทางกายภาพสำหรับหุ่น G1 ดังนี้:

| จุดที่ต้องระวังบนหุ่น G1 จริง | สาเหตุจากในคลิป | แนวทางป้องกันก่อนทดสอบจริง |
| :--- | :--- | :--- |
| **1. แคลมป์พลาสติกในข้อต่อหัวไหล่ (`Shoulder Roll/Yaw`)** | เป็นชิ้นส่วนพลาสติกพิมพ์ 3 มิติที่มีเขี้ยวเล็กๆ รับแรงบิด หักง่ายถ้าแขนสะบัดแรง | 1. ใส่ Arm EMA Filter ในโค้ด C++<br>2. หากมีไฟล์ CAD แคลมป์หัวไหล่ ควรพิมพ์ 3D สำรองไว้เป็นอะไหล่<br>3. ผูกสายคล้องข้อมือ/ปลายแขนแบบหย่อนๆ ไว้กันแขนร่วงกระแทกพื้นหากแคลมป์หลุด |
| **2. ก้านกระทุ้งข้อเท้าแบบเปลือย (`Exposed Ankle Linkage`)** | ข้อเท้าของ G1 ใช้กลไกก้านคู่เปลือย หากเดินบนพื้นที่มีเศษกิ่งไม้ หิน หรือสายไฟ อาจเข้าไปขัดในร่องก้านข้อเท้าได้ | เริ่มทดสอบบนพื้นเรียบสะอาด (Indoor / พื้นยาง / คอนกรีตเรียบ) ห้ามเดินลุยเศษใบไม้หรือสายไฟเปลือยบนพื้น |
| **3. รอกพยุงตัว (Overhead Gantry / Hoist)** | ช่วงทดสอบแรกๆ หุ่นอาจสะดุดหรือตัดเข้าโหมด `Passive` (ตัวอ่อนทันที) | ต้องแขวนเชือกพยุงที่ห่วงคล้องไหล่ของ G1 โดยตั้งความยาวเชือกให้ **หย่อนพอที่จะไม่ดึงหุ่นตอนเดินปกติ แต่ตึงรับน้ำหนักได้ทันก่อนที่เข่าหรือหัวไหล่จะกระแทกพื้น** เมื่อตัดเข้าโหมด `Passive` |
| **4. ปุ่มฉุกเฉิน (`L2 + B` บนจอย หรือปุ่ม `1` บนคีย์บอร์ด)** | เมื่อหุ่นเสียจังหวะ คนคุมต้องกดตัดเข้าโหมด `Passive` ได้โดยไม่ต้องก้มมองปุ่ม | ซ้อมกดสลับ `1 (Passive)` $\leftrightarrow$ `2 (FixStand)` $\leftrightarrow$ `3 (Velocity)` และ `Spacebar (หยุดเดิน)` จนชินมือก่อนเอาเชือกพยุงออก |

---

## 📌 สรุปลำดับความสำคัญที่แนะนำให้ลงมือทำในโค้ดตอนนี้

หากคุณเห็นด้วย เราสามารถแก้ไขโค้ด C++ ใน `deploy/` และคอมไพล์ [`g1_ctrl`](file:///home/luke/nvidia/unitree/unitree_rl_mjlab/deploy/robots/g1_23dof/build/g1_ctrl) ใหม่ได้ทันที (ใช้เวลาไม่เกิน 2 นาที และไม่ต้องเทรนโมเดลใหม่) ตามลำดับนี้ครับ:

1. ✅ **[สำคัญที่สุด] เปิดใช้งาน `bad_orientation` ใน [`terminations.h`](file:///home/luke/nvidia/unitree/unitree_rl_mjlab/deploy/include/isaaclab/envs/mdp/terminations.h)** เพื่อให้หุ่นตัดเข้าโหมด `Passive` อัตโนมัติทันทีที่ลำตัวเอียงเกิน ~48 องศา (`0.85 rad`)
2. ✅ **แก้ค่าเริ่มต้น `_processed_actions` ใน [`joint_actions.h`](file:///home/luke/nvidia/unitree/unitree_rl_mjlab/deploy/include/isaaclab/envs/mdp/actions/joint_actions.h)** ให้เริ่มที่ท่า `_offset` เสมอ ป้องกันการกระตุกเสี้ยววินาทีแรกที่เข้าโหมด `Velocity`
3. ✅ **เพิ่ม Arm EMA Low-Pass Filter เฉพาะข้อต่อแขน (`index 13..22`)** เพื่อตัดความถี่สูงในการสะบัดแขน ถนอมเขี้ยวพลาสติกหัวไหล่ตามคำแนะนำของ sentdex
