document.addEventListener("DOMContentLoaded", function () {
    const canvas = document.getElementById('sphere-canvas');
    const ctx = canvas.getContext('2d');
    const numSpheres = 10;
    const spheres = [];
    let frameCount = 0;

    function drawSphere(x, y, radius) {
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(255, 0, 0, 0.5)';
        ctx.fill();
        ctx.closePath();
    }

    function animate() {
        requestAnimationFrame(animate);

        ctx.clearRect(0, 0, canvas.width, canvas.height);

        if (frameCount < numSpheres) {
            for (let i = 0; i < frameCount; i++) {
                const x = i * 80 + 50;  // Adjust the positioning as needed
                const y = 300;
                const radius = 20;
                drawSphere(x, y, radius);
            }
            frameCount++;
        }
    }

    animate();
});
