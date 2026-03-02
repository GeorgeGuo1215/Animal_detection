/**
 * 姿态解算模块 - Attitude Solver
 * 支持 Madgwick 滤波器和互补滤波器
 */

class AttitudeSolver {
    constructor() {
        // 四元数 [w, x, y, z]
        this.q = [1, 0, 0, 0];

        // 欧拉角 (弧度)
        this.pitch = 0;  // 俯仰角
        this.roll = 0;   // 横滚角
        this.yaw = 0;    // 偏航角

        // 角速度 (rad/s)
        this.gx = 0;
        this.gy = 0;
        this.gz = 0;

        // 算法选择: 'madgwick' 或 'complementary'
        this.algorithm = 'madgwick';

        // Madgwick 滤波器参数 (增大 beta 提高响应速度)
        this.beta = 0.5;  // 增益参数 (0.1-1.0, 越大越灵敏但可能不稳定)

        // 互补滤波器参数
        this.alpha = 0.96;  // 陀螺仪权重 (降低以增加加速度计影响)

        // 采样时间 (秒)
        this.dt = 0.01;  // 默认 100Hz

        // 上次更新时间
        this.lastUpdateTime = null;
    }

    /**
     * 设置算法类型
     */
    setAlgorithm(algorithm) {
        if (algorithm === 'madgwick' || algorithm === 'complementary') {
            this.algorithm = algorithm;
            console.log(`✅ 姿态解算算法切换为: ${algorithm}`);
        }
    }

    /**
     * 设置采样频率
     */
    setSampleRate(fs) {
        this.dt = 1.0 / fs;
    }

    /**
     * 更新姿态 - 主入口
     * @param {number} gx - 陀螺仪 X 轴 (deg/s)
     * @param {number} gy - 陀螺仪 Y 轴 (deg/s)
     * @param {number} gz - 陀螺仪 Z 轴 (deg/s)
     * @param {number} ax - 加速度计 X 轴 (g)
     * @param {number} ay - 加速度计 Y 轴 (g)
     * @param {number} az - 加速度计 Z 轴 (g)
     */
    update(gx, gy, gz, ax, ay, az) {
        // 计算实际采样时间
        const now = Date.now();
        if (this.lastUpdateTime) {
            this.dt = (now - this.lastUpdateTime) / 1000.0;
            // 限制 dt 范围，避免异常值
            this.dt = Math.max(0.001, Math.min(this.dt, 0.1));
        } else {
            this.dt = 0.01; // 默认 100Hz
        }
        this.lastUpdateTime = now;

        // 保存角速度 (转换为 rad/s)
        this.gx = gx * Math.PI / 180;
        this.gy = gy * Math.PI / 180;
        this.gz = gz * Math.PI / 180;

        // 根据算法选择更新四元数
        if (this.algorithm === 'madgwick') {
            this.updateMadgwick(this.gx, this.gy, this.gz, ax, ay, az);
        } else {
            this.updateComplementary(this.gx, this.gy, this.gz, ax, ay, az);
        }

        // 从四元数计算欧拉角
        this.quaternionToEuler();
    }

    /**
     * Madgwick 滤波器更新
     */
    updateMadgwick(gx, gy, gz, ax, ay, az) {
        let [q0, q1, q2, q3] = this.q;

        // 归一化加速度计数据
        const norm = Math.sqrt(ax * ax + ay * ay + az * az);
        if (norm === 0) return;
        ax /= norm;
        ay /= norm;
        az /= norm;

        // 目标函数的梯度
        const s0 = -2 * q2 * (2 * q1 * q3 - 2 * q0 * q2 - ax) +
                   -2 * q1 * (2 * q0 * q1 + 2 * q2 * q3 - ay) +
                   0;
        const s1 = 2 * q3 * (2 * q1 * q3 - 2 * q0 * q2 - ax) +
                   2 * q0 * (2 * q0 * q1 + 2 * q2 * q3 - ay) +
                   -4 * q1 * (1 - 2 * q1 * q1 - 2 * q2 * q2 - az);
        const s2 = -2 * q0 * (2 * q1 * q3 - 2 * q0 * q2 - ax) +
                   2 * q3 * (2 * q0 * q1 + 2 * q2 * q3 - ay) +
                   -4 * q2 * (1 - 2 * q1 * q1 - 2 * q2 * q2 - az);
        const s3 = 2 * q1 * (2 * q1 * q3 - 2 * q0 * q2 - ax) +
                   2 * q2 * (2 * q0 * q1 + 2 * q2 * q3 - ay) +
                   0;

        // 归一化梯度
        const sNorm = Math.sqrt(s0 * s0 + s1 * s1 + s2 * s2 + s3 * s3);
        const qs0 = s0 / sNorm;
        const qs1 = s1 / sNorm;
        const qs2 = s2 / sNorm;
        const qs3 = s3 / sNorm;

        // 四元数导数
        const qDot0 = 0.5 * (-q1 * gx - q2 * gy - q3 * gz) - this.beta * qs0;
        const qDot1 = 0.5 * (q0 * gx + q2 * gz - q3 * gy) - this.beta * qs1;
        const qDot2 = 0.5 * (q0 * gy - q1 * gz + q3 * gx) - this.beta * qs2;
        const qDot3 = 0.5 * (q0 * gz + q1 * gy - q2 * gx) - this.beta * qs3;

        // 积分四元数
        q0 += qDot0 * this.dt;
        q1 += qDot1 * this.dt;
        q2 += qDot2 * this.dt;
        q3 += qDot3 * this.dt;

        // 归一化四元数
        const qNorm = Math.sqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3);
        this.q = [q0 / qNorm, q1 / qNorm, q2 / qNorm, q3 / qNorm];
    }

    /**
     * 互补滤波器更新
     */
    updateComplementary(gx, gy, gz, ax, ay, az) {
        // 归一化加速度计数据
        const norm = Math.sqrt(ax * ax + ay * ay + az * az);
        if (norm === 0) return;
        ax /= norm;
        ay /= norm;
        az /= norm;

        // 从加速度计计算俯仰角和横滚角
        const pitchAcc = Math.atan2(ax, Math.sqrt(ay * ay + az * az));
        const rollAcc = Math.atan2(ay, az);

        // 陀螺仪积分
        const pitchGyro = this.pitch + gy * this.dt;
        const rollGyro = this.roll + gx * this.dt;
        const yawGyro = this.yaw + gz * this.dt;

        // 互补滤波
        this.pitch = this.alpha * pitchGyro + (1 - this.alpha) * pitchAcc;
        this.roll = this.alpha * rollGyro + (1 - this.alpha) * rollAcc;
        this.yaw = yawGyro;  // 偏航角只能用陀螺仪

        // 从欧拉角转换为四元数
        this.eulerToQuaternion();
    }

    /**
     * 四元数转欧拉角
     */
    quaternionToEuler() {
        const [q0, q1, q2, q3] = this.q;

        // Roll (x-axis rotation)
        const sinr_cosp = 2 * (q0 * q1 + q2 * q3);
        const cosr_cosp = 1 - 2 * (q1 * q1 + q2 * q2);
        this.roll = Math.atan2(sinr_cosp, cosr_cosp);

        // Pitch (y-axis rotation)
        const sinp = 2 * (q0 * q2 - q3 * q1);
        if (Math.abs(sinp) >= 1) {
            this.pitch = Math.sign(sinp) * Math.PI / 2;
        } else {
            this.pitch = Math.asin(sinp);
        }

        // Yaw (z-axis rotation)
        const siny_cosp = 2 * (q0 * q3 + q1 * q2);
        const cosy_cosp = 1 - 2 * (q2 * q2 + q3 * q3);
        this.yaw = Math.atan2(siny_cosp, cosy_cosp);
    }

    /**
     * 欧拉角转四元数
     */
    eulerToQuaternion() {
        const cy = Math.cos(this.yaw * 0.5);
        const sy = Math.sin(this.yaw * 0.5);
        const cp = Math.cos(this.pitch * 0.5);
        const sp = Math.sin(this.pitch * 0.5);
        const cr = Math.cos(this.roll * 0.5);
        const sr = Math.sin(this.roll * 0.5);

        this.q[0] = cr * cp * cy + sr * sp * sy;
        this.q[1] = sr * cp * cy - cr * sp * sy;
        this.q[2] = cr * sp * cy + sr * cp * sy;
        this.q[3] = cr * cp * sy - sr * sp * cy;
    }

    /**
     * 获取欧拉角 (度)
     */
    getEulerAngles() {
        return {
            pitch: this.pitch * 180 / Math.PI,
            roll: this.roll * 180 / Math.PI,
            yaw: this.yaw * 180 / Math.PI
        };
    }

    /**
     * 获取四元数
     */
    getQuaternion() {
        return {
            w: this.q[0],
            x: this.q[1],
            y: this.q[2],
            z: this.q[3]
        };
    }

    /**
     * 获取角速度 (deg/s)
     */
    getAngularVelocity() {
        return {
            gx: this.gx * 180 / Math.PI,
            gy: this.gy * 180 / Math.PI,
            gz: this.gz * 180 / Math.PI
        };
    }

    /**
     * 重置姿态
     */
    reset() {
        this.q = [1, 0, 0, 0];
        this.pitch = 0;
        this.roll = 0;
        this.yaw = 0;
        this.lastUpdateTime = null;
    }
}

/**

 * 3D 姿态可视化器
 * 使用 Three.js 显示立方体姿态
 */
class AttitudeVisualizer {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        if (!this.container) {
            console.error('❌ 找不到容器:', containerId);
            return;
        }

        // 初始化 Three.js
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.cube = null;
        this.animationId = null;

        this.init();
    }

    /**
     * 初始化 3D 场景
     */
    init() {
        console.log('🎯 开始初始化 3D 场景...');
        console.log('THREE 对象:', typeof THREE);

        if (typeof THREE === 'undefined') {
            console.error('❌ THREE.js 未加载!');
            this.container.innerHTML = '<div style="color:#ff6b6b;text-align:center;padding:40px;">Three.js 未加载，请检查网络连接</div>';
            return;
        }

        // 创建场景
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x1a1a2e);
        console.log('✅ 场景创建成功');

        // 创建相机 — 使用实际尺寸，若容器仍为0则用合理默认值
        const width = this.container.clientWidth || this.container.offsetWidth || 600;
        const height = this.container.clientHeight || this.container.offsetHeight || 400;
        console.log(`📐 容器尺寸: ${width}x${height}`);

        if (width === 0 || height === 0) {
            console.error('❌ 容器尺寸为 0!');
            return;
        }

        this.camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
        this.camera.position.set(4, 4, 4);
        this.camera.lookAt(0, 0, 0);
        console.log('✅ 相机创建成功');

        // 创建渲染器
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(width, height);
        this.container.appendChild(this.renderer.domElement);
        console.log('✅ 渲染器创建成功');

        // 添加环境光
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        this.scene.add(ambientLight);

        // 添加方向光
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(5, 5, 5);
        this.scene.add(directionalLight);
        console.log('✅ 光照添加成功');

        // 创建立方体
        const geometry = new THREE.BoxGeometry(2, 1, 0.5);
        const materials = [
            new THREE.MeshLambertMaterial({ color: 0xff0000 }), // 右 - 红色
            new THREE.MeshLambertMaterial({ color: 0x00ff00 }), // 左 - 绿色
            new THREE.MeshLambertMaterial({ color: 0x0000ff }), // 上 - 蓝色
            new THREE.MeshLambertMaterial({ color: 0xffff00 }), // 下 - 黄色
            new THREE.MeshLambertMaterial({ color: 0xff00ff }), // 前 - 品红
            new THREE.MeshLambertMaterial({ color: 0x00ffff })  // 后 - 青色
        ];
        this.cube = new THREE.Mesh(geometry, materials);
        this.scene.add(this.cube);
        console.log('✅ 立方体创建成功');

        // 添加坐标轴辅助线
        const axesHelper = new THREE.AxesHelper(2);
        this.scene.add(axesHelper);

        // 添加网格
        const gridHelper = new THREE.GridHelper(10, 10, 0x444444, 0x222222);
        this.scene.add(gridHelper);

        // 开始渲染
        this.animate();
        console.log('✅ 3D 场景初始化完成!');

        // 窗口大小调整
        window.addEventListener('resize', () => this.onWindowResize());
    }

    /**
     * 更新立方体姿态
     * @param {number} pitch - 俯仰角 (弧度)
     * @param {number} roll - 横滚角 (弧度)
     * @param {number} yaw - 偏航角 (弧度)
     */
    updateAttitude(pitch, roll, yaw) {
        if (!this.cube) return;

        // 设置旋转 (注意 Three.js 使用 ZYX 顺序)
        this.cube.rotation.set(pitch, yaw, roll);
    }

    /**
     * 使用四元数更新姿态
     */
    updateQuaternion(w, x, y, z) {
        if (!this.cube) return;
        this.cube.quaternion.set(x, y, z, w);
    }

    /**
     * 动画循环
     */
    animate() {
        if (!this.renderer || !this.scene || !this.camera) return;
        this.animationId = requestAnimationFrame(() => this.animate());
        this.renderer.render(this.scene, this.camera);
    }

    /**
     * 窗口大小调整
     */
    onWindowResize() {
        if (!this.container || !this.camera || !this.renderer) return;

        const width = this.container.clientWidth;
        const height = this.container.clientHeight || 400;

        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height);
    }

    /**
     * 停止渲染
     */
    stop() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }
    }

    /**
     * 重新启动渲染
     */
    start() {
        if (!this.animationId && this.renderer) {
            this.animate();
        }
    }

    /**
     * 销毁
     */
    destroy() {
        this.stop();
        if (this.renderer && this.container) {
            this.container.removeChild(this.renderer.domElement);
        }
    }
}
