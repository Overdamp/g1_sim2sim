# G1 Sim-to-Sim

โครงการฝึก G1 ให้ยืน/เดิน แล้วทดสอบ controller ผ่าน Unitree SDK2/DDS กับ simulator ก่อนเตรียม sim-to-real

**สถานะ: scaffold เท่านั้น** ยังไม่มี custom task ที่ลงทะเบียน, controller executable หรือผล training/deployment ที่ผ่านการทดสอบ Script ที่ยังไม่ implement จะแจ้งสถานะและจบด้วย exit code 2 โดยไม่เปิด DDS หรือสั่ง motor

## เริ่มอ่าน

คำสั่งสำหรับเปิดหุ่นเปล่า, อ่าน LowState และทดสอบส่งคำสั่งเล็กน้อยผ่าน ROS 2 ใน simulator: [เปิด G1 simulation + อ่าน/ส่ง Unitree DDS](docs/run_empty_simulation.md)

สำหรับศึกษา environment ทีละส่วน เริ่มที่ [ตาราง Scene → Commands → Actions → Observations → Rewards → Terminations → Events → Curriculum](docs/environment_components.md)

อ่านคู่กับ [ตารางฝั่งหุ่นจริง Unitree G1](docs/real_robot_components_unitree.md) สำหรับ SDK messages, motor mapping, observations และขั้นตอนทดสอบ hardware

1. [แผน implement](docs/implementation_plan.md)
2. [สถาปัตยกรรม](docs/architecture.md)
3. [การติดตั้งและตรวจ baseline](docs/setup.md)
4. [Training](docs/training.md), [Sim-to-sim](docs/sim2sim.md), [Evaluation](docs/evaluation.md)
5. [ดัชนีเอกสารเดิม](docs/reference/README.md)

## โครงสร้าง

| Directory | หน้าที่ |
| --- | --- |
| `configs/` | Robot, transport, timing, fault policy และ scenarios; ยังไม่ validated |
| `src/g1_sim2sim/` | Python tasks, export และ evaluation modules |
| `controller/` | จุดเริ่ม C++ controller; ไม่มี runtime target ในขั้นนี้ |
| `scripts/` | CLI entry points ที่รอ implementation |
| `tests/` | ขอบเขต contract/parity/fault tests ที่จะพัฒนา |
| `manifests/` | Dependency paths, versions และ migration record |
| `artifacts/` | Immutable exported policy bundles |
| `runs/` | Config snapshot, checkpoints, logs และ metrics ต่อ run |
| `build/` | Generated C++ build output |

ฐานหลักคือ [unitree_rl_mjlab](../unitree_rl_mjlab/README.md) ซึ่ง pin mjlab 1.2.0 แยก environment จาก standalone mjlab 1.6.0 ไม่มีการติดตั้ง dependencies หรือแก้ upstream ในการสร้าง scaffold ครั้งนี้

## ลำดับเริ่มพัฒนา

- [ ] I01 ยืนยัน dependency revisions และ robot variant
- [ ] I02 build/run upstream baseline ผ่าน DDS
- [ ] I03 เพิ่ม configurable domain/interface ใน controller
- [ ] I04 dump/validate joint mapping
- [ ] I05 กำหนด actor observation/action contract

ดูรายละเอียด I01–I17 ในแผนหลัก ห้ามนับการสร้าง placeholder ว่างานเหล่านี้ผ่านแล้ว

## การจัดเก็บ

เก็บ source/config/docs และ manifest ใน Git ส่วน checkpoints, ONNX, logs, videos และ build output ให้ backup แยก ไม่แก้ทับ artifact version ที่ใช้ประเมินแล้ว
