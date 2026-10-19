# Evaluation

สถานะ: scenario YAML เป็นข้อกำหนดเริ่มต้น ยังไม่มี harness ประมวลผล

ลำดับ: schema/joint mapping → observation/ONNX/action parity → DDS → PD → stand → walk → stop/fault → dynamics sweep

บันทึก fall rate, survival, drift, velocity RMSE, target/actual error, saturation, inference timing, state age และ fault detection latency

Evaluation ที่วัดการยืนเดินต้องไม่มี elastic band/support force แยก timeout ออกจาก fall termination เก็บ sim time และ wall time คนละ field

รายงานทุกครั้งพร้อม seed, model/policy/config hashes, scenario และจำนวน trials ดู [แผนหลัก](implementation_plan.md) Phase 8 และเกณฑ์ผ่าน
