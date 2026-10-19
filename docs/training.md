# Training

สถานะ: `tasks/stand.py`, `walk.py`, `rl_cfg.py` เป็น module placeholders ยังไม่ register task

- ใช้ฐาน Unitree task `Unitree-G1-Flat` ใน environment ของ unitree_rl_mjlab
- กำหนด robot/observation/action contract ก่อนสร้าง custom tasks
- เริ่ม zero-command stand และ forward walk ช่วงแคบพร้อม stop commands
- เก็บ config/source/version snapshots ต่อ run ใน `runs/`
- Export deterministic actor และ normalization พร้อม deploy configuration ของ run จริง

[คู่มือเรียน RL เดิม](reference/training_from_scratch_mjlab.md) ใช้ standalone mjlab คนละเวอร์ชัน ตัวอย่างไม่ใช่คำสั่งพร้อมรันใน scaffold นี้

รายละเอียดอยู่ใน [แผนหลัก](implementation_plan.md) Phase 2–4
