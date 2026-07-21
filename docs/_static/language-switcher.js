(function () {
  "use strict";

  function installLanguageSwitcher() {
    var path = window.location.pathname;
    var isChinese = path === "/zh_CN" || path.indexOf("/zh_CN/") === 0;
    var target = isChinese
      ? path.replace(/^\/zh_CN(?=\/|$)/, "") || "/"
      : "/zh_CN" + (path.charAt(0) === "/" ? path : "/" + path);

    var searchArea = document.querySelector(".wy-side-nav-search");
    if (searchArea && !document.querySelector(".pyhermes-language-switch")) {
      var link = document.createElement("a");
      link.className = "pyhermes-language-switch";
      link.href = target + window.location.search + window.location.hash;
      link.lang = isChinese ? "en" : "zh-CN";
      link.textContent = isChinese ? "English" : "简体中文";
      searchArea.appendChild(link);
    }

    if (isChinese) {
      var content = document.querySelector(".wy-nav-content");
      if (content && !document.querySelector(".pyhermes-translation-notice")) {
        var notice = document.createElement("div");
        notice.className = "pyhermes-translation-notice";
        notice.textContent =
          "简体中文文档正在逐步完善；尚未翻译的页面会暂时显示英文原文。";
        content.insertBefore(notice, content.firstChild);
      }
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", installLanguageSwitcher);
  } else {
    installLanguageSwitcher();
  }
})();
