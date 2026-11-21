// Thumbs up/down: click to vote, click the active one again to clear.
(function () {
  const csrf = () => document.querySelector("input[name=csrfmiddlewaretoken]")?.value || "";

  document.addEventListener("click", async (ev) => {
    const btn = ev.target.closest(".thumb");
    if (!btn) return;
    if (!window.MV_LOGGED_IN) {
      location.href = (window.MV_LOGIN || "/accounts/login/") + "?next=" + encodeURIComponent(location.pathname + location.search);
      return;
    }
    const box = btn.closest(".vote");
    const holder = btn.closest("[data-track]");
    const value = btn.classList.contains("active") ? 0 : Number(btn.dataset.value);
    const body = new URLSearchParams({ recording_mbid: holder.dataset.track, value, query: box.dataset.query || "" });
    box.classList.add("busy");
    try {
      const r = await fetch("/feedback/", {
        method: "POST", body, headers: { "X-CSRFToken": csrf() }, credentials: "same-origin",
      });
      if (!r.ok) throw new Error(await r.text());
      box.querySelectorAll(".thumb").forEach((b) => b.classList.remove("active"));
      if (value !== 0) btn.classList.add("active");
      holder.classList.toggle("liked", value === 1);
      holder.classList.toggle("disliked", value === -1);
    } catch (e) {
      console.error(e);
      box.classList.add("err");
      setTimeout(() => box.classList.remove("err"), 1200);
    } finally {
      box.classList.remove("busy");
    }
  });
})();
