// Theme Switcher
(function() {
  'use strict';

  // Get saved theme from localStorage or default to 'light'
  const savedTheme = localStorage.getItem('roster-theme') || 'light';

  // Apply theme on page load
  function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('roster-theme', theme);
  }

  // Initialize theme
  applyTheme(savedTheme);

  // Wait for DOM to be ready
  document.addEventListener('DOMContentLoaded', function() {
    // Create theme selector
    const themeSelector = document.createElement('div');
    themeSelector.className = 'theme-selector';
    themeSelector.innerHTML = `
      <label for="theme-select">Theme:</label>
      <select id="theme-select">
        <option value="light">Light</option>
        <option value="dark">Dark</option>
        <option value="blue">Blue</option>
        <option value="green">Green</option>
      </select>
    `;
    document.body.appendChild(themeSelector);

    // Get theme select element
    const themeSelect = document.getElementById('theme-select');
    
    // Set current theme in select
    themeSelect.value = savedTheme;

    // Handle theme change
    themeSelect.addEventListener('change', function(e) {
      const newTheme = e.target.value;
      applyTheme(newTheme);
    });
  });
})();
