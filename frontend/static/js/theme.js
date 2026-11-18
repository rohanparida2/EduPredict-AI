(() => {
    const styles = getComputedStyle(document.documentElement);
    window.eduPredictTheme = {
        red: styles.getPropertyValue('--color-red').trim(),
        redDark: styles.getPropertyValue('--color-red-dark').trim(),
        purple: styles.getPropertyValue('--color-purple').trim(),
        light: styles.getPropertyValue('--color-card').trim(),
        white: styles.getPropertyValue('--color-white').trim(),
        muted: styles.getPropertyValue('--color-axis').trim(),
        grid: styles.getPropertyValue('--color-grid').trim(),
    };
})();
