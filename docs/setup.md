# Setup

สถานะ: ยังไม่ได้ติดตั้ง dependencies/build/run baseline สำหรับโครงการนี้

1. ตรวจ `manifests/dependencies.yaml` และยืนยัน commit ที่ใช้
2. สร้าง Python 3.11 environment แยก ตาม [setup ของฐาน](../../unitree_rl_mjlab/doc/setup_en.md)
3. ติดตั้ง `unitree_rl_mjlab` แบบ editable ใน environment นั้นตามคู่มือฐาน
4. จาก root ของ `g1_sim2sim` ใช้ `python -m pip install -e .` เพื่อติดตั้ง package scaffold ด้วย Python ของ environment เดียวกัน
5. Build baseline simulator/controller ตาม [implementation plan](implementation_plan.md) Phase 1 และบันทึกผล

`pyproject.toml` ยังไม่ประกาศ runtime dependencies เพราะต้องยืนยัน dependency set ของ baseline ก่อน ไม่ได้หมายความว่า training/controller ใช้ standard library อย่างเดียว

Standalone `../../../mjlab/` เป็น framework อีกเวอร์ชัน ไม่ติดตั้งทับ environment นี้

ตรวจ scaffold โดยไม่ติดตั้ง dependency:

```bash
python3 scripts/train.py --help
cmake -S . -B build
```

CMake configure เพียงตรวจ layout; ยังไม่มี controller target คำสั่ง train/play ที่ไม่มี `--help` จะจบ exit code 2 จนกว่าจะ implement
