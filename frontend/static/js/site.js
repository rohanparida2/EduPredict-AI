(() => {
    const drawer = document.querySelector('[data-nav-drawer]');
    const backdrop = document.querySelector('[data-drawer-backdrop]');
    const menuButton = document.querySelector('[data-menu-toggle]');
    const closeButton = document.querySelector('[data-menu-close]');
    const searchToggle = document.querySelector('[data-search-toggle]');
    const searchForm = document.querySelector('[data-search-form]');
    const searchInput = document.querySelector('[data-search-input]');

    const setDrawerOpen = (isOpen) => {
        if (!drawer || !backdrop || !menuButton) return;
        drawer.classList.toggle('is-open', isOpen);
        backdrop.classList.toggle('is-open', isOpen);
        drawer.setAttribute('aria-hidden', String(!isOpen));
        drawer.toggleAttribute('inert', !isOpen);
        menuButton.setAttribute('aria-expanded', String(isOpen));
        document.body.classList.toggle('drawer-open', isOpen);
        if (isOpen) closeButton?.focus();
    };

    menuButton?.addEventListener('click', () => {
        setDrawerOpen(menuButton.getAttribute('aria-expanded') !== 'true');
    });
    closeButton?.addEventListener('click', () => setDrawerOpen(false));
    backdrop?.addEventListener('click', () => setDrawerOpen(false));
    document.addEventListener('keydown', (event) => {
        if (event.key === 'Escape') {
            setDrawerOpen(false);
            searchForm?.classList.remove('is-open');
            searchToggle?.setAttribute('aria-expanded', 'false');
        }
    });

    searchToggle?.addEventListener('click', () => {
        const isOpen = searchForm?.classList.toggle('is-open') ?? false;
        searchToggle.setAttribute('aria-expanded', String(isOpen));
        if (isOpen) searchInput?.focus();
        else {
            if (searchInput) searchInput.value = '';
            filterSearch('');
        }
    });

    function filterSearch(query) {
        const term = query.trim().toLocaleLowerCase();
        document.querySelectorAll('[data-searchable]').forEach((item) => {
            item.hidden = Boolean(term) && !item.textContent.toLocaleLowerCase().includes(term);
        });
    }

    searchInput?.addEventListener('input', () => filterSearch(searchInput.value));
    searchForm?.addEventListener('submit', (event) => event.preventDefault());

    document.querySelectorAll('[data-view-toggle]').forEach((toggle) => {
        const target = document.querySelector(toggle.dataset.viewToggle);
        if (!target) return;
        toggle.querySelectorAll('[data-view]').forEach((button) => {
            button.addEventListener('click', () => {
                const listView = button.dataset.view === 'list';
                target.classList.toggle('is-list', listView);
                toggle.querySelectorAll('[data-view]').forEach((item) => {
                    item.setAttribute('aria-pressed', String(item === button));
                });
            });
        });
    });

})();
