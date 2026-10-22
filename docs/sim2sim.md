# Sim-to-Sim

เริ่มรันแบบไม่ใช้ policy ได้จาก [คู่มือคำสั่งเปิด G1 เปล่าและตรวจ DDS](run_empty_simulation.md)

สถานะ: ยังไม่ได้ run DDS baseline หรือ custom controller

1. ตรวจ robot model และ sensor/joint mapping
2. ใช้ integrated simulator และ G1 controller ของ `unitree_rl_mjlab` เป็น baseline
3. Baseline ปัจจุบันใช้ domain 0 + `lo`; `configs/transport/sim.yaml` บันทึกคู่นี้
4. เมื่อเพิ่ม domain option ให้ controller แล้วจึงทดสอบ profile domain 1 ทั้งสองฝั่ง
5. เริ่ม read-only LowState → actuator mapping ใน sim → FixStand → policy stand → walk
6. ตรวจ parity และ faults ก่อนรายงานผลสำเร็จ

ค่าที่อยู่ใน config ของโครงการยังไม่ได้เชื่อมต่อกับ upstream launcher/controller โดยอัตโนมัติ

ดูคำสั่ง build/run ที่มีอยู่จริงใน [แผนหลัก](implementation_plan.md) Phase 1
