import requests
import json
import hashlib
import re

from bs4 import BeautifulSoup
from protego import Protego
from urllib.parse import urlparse, urljoin
from datetime import datetime, timezone


USER_AGENT = "NetVoraNewsBot/1.0 (+https://netvora.org/)"

HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "es-BO,es;q=0.9"
}

session = requests.Session()
session.headers.update(HEADERS)

robots_cache = {}

ALLOWED_DOMAINS = {
    "t.co",
    "eldeber.com.bo",
    "unitel.bo",
    "lostiempos.com",
    "correodelsur.com",
    "la-razon.com",
    "reduno.com.bo",
    "eldia.com.bo",
    "opinion.com.bo",
    "erbol.com.bo",
    "vision360.bo",
    "brujuladigital.net",
    "urgente.bo",
    "elpais.bo",
    "noticiasfides.com"
}


def validar_url(url):

    parsed = urlparse(url)

    domain = parsed.hostname or ""

    if (
        parsed.scheme not in ("http", "https")
        or parsed.username
        or parsed.password
        or parsed.port not in (None, 80, 443)
    ):
        raise ValueError("INVALID_URL")

    if not any(
        domain == d or domain.endswith("." + d)
        for d in ALLOWED_DOMAINS
    ):
        raise ValueError("DOMAIN_NOT_ALLOWED")

    return parsed


def verificar_robots(url):

    parsed = validar_url(url)

    origin = f"{parsed.scheme}://{parsed.netloc}"

    if origin not in robots_cache:

        robots_url = origin + "/robots.txt"

        with session.get(
            robots_url,
            timeout=10,
            allow_redirects=False
        ) as response:

            if response.status_code == 200:

                if response.text.lstrip().startswith("<"):
                    raise ValueError("ROBOTS_UNAVAILABLE")

                robots_cache[origin] = Protego.parse(
                    response.text
                )

            elif response.status_code in (404, 410):

                robots_cache[origin] = Protego.parse("")

            else:

                raise ValueError("ROBOTS_UNAVAILABLE")

    return robots_cache[origin].can_fetch(
        url,
        "NetVoraNewsBot"
    )


def descargar_web(url):

    # Resolver redirecciones

    for _ in range(10):

        parsed = validar_url(url)

        if not verificar_robots(url):

            if parsed.hostname == "t.co":
                raise ValueError("NEEDS_EXPANDED_URL")

            raise ValueError("ROBOTS_DISALLOWED")

        with session.get(
            url,
            timeout=15,
            allow_redirects=False,
            stream=True
        ) as response:

            if response.status_code in (
                301, 302, 303, 307, 308
            ):

                location = response.headers.get("Location")

                if not location:
                    raise ValueError("INVALID_REDIRECT")

                url = urljoin(url, location)

                continue

            if response.status_code != 200:

                raise ValueError(
                    f"HTTP_{response.status_code}"
                )

            if "html" not in response.headers.get(
                "Content-Type", ""
            ).lower():

                raise ValueError("NOT_HTML")

            html = bytearray()

            for chunk in response.iter_content(65536):

                html.extend(chunk)

                if len(html) > 8 * 1024 * 1024:

                    raise ValueError("RESPONSE_TOO_LARGE")

            return url, bytes(html)

    raise ValueError("TOO_MANY_REDIRECTS")


def obtener_meta(soup, name):

    tag = soup.find(
        "meta",
        attrs={"property": name}
    )

    if not tag:

        tag = soup.find(
            "meta",
            attrs={"name": name}
        )

    if not tag:

        tag = soup.find(
            "meta",
            attrs={"itemprop": name}
        )

    return tag.get("content") if tag else None


def extraer_contenido(soup):

    # Eliminar elementos innecesarios

    for tag in soup.select(
        "script, style, nav, footer, header, "
        "aside, iframe, form, noscript, button, "
        ".advertisement, .publicidad, .social-share, "
        ".related-news, .related-posts, .comments, "
        ".noticias-relacionadas, .newsletter"
    ):

        tag.decompose()

    # Selectores habituales de medios de noticias

    selectores = [
        '[itemprop="articleBody"]',
        ".article-body",
        ".article-content",
        ".article-text",
        ".article__body",
        ".article__content",
        ".news-content",
        ".news-body",
        ".entry-content",
        ".post-content",
        ".post-body",
        ".story-body",
        ".story-content",
        ".nota-contenido",
        ".contenido-noticia",
        ".field-name-body",
        ".field--name-body",
        "article",
        "main"
    ]

    mejor_contenido = ""
    mejor_selector = None

    for selector in selectores:

        for container in soup.select(selector):

            parrafos = []

            for p in container.find_all("p"):

                texto = p.get_text(
                    separator=" ",
                    strip=True
                )

                texto = re.sub(
                    r"\s+",
                    " ",
                    texto
                )

                if len(texto) >= 30:

                    parrafos.append(texto)

            # Eliminar parrafos duplicados

            parrafos = list(
                dict.fromkeys(parrafos)
            )

            contenido = "\n\n".join(parrafos)

            if len(contenido) > len(mejor_contenido):

                mejor_contenido = contenido
                mejor_selector = selector

        # Priorizar contenedores especificos

        if len(mejor_contenido.split()) >= 80:

            return mejor_contenido, mejor_selector

    return mejor_contenido, mejor_selector


def scrape_article(url, expanded_url=None):

    result = {
        "ok": False,
        "original_url": url,
        "final_url": None,
        "canonical_url": None,
        "domain": None,
        "title": None,
        "description": None,
        "content": None,
        "author": None,
        "published_at": None,
        "updated_at": None,
        "image_url": None,
        "category": None,
        "word_count": 0,
        "content_hash": None,
        "scraping_status": "PENDING",
        "scraping_method": "BEAUTIFULSOUP",
        "selector_used": None,
        "scraped_at": None,
        "error": None
    }

    try:

        # Priorizar URL original proporcionada por X

        target_url = expanded_url or url

        final_url, html = descargar_web(target_url)

        result["final_url"] = final_url
        result["domain"] = urlparse(final_url).hostname

        soup = BeautifulSoup(html, "html.parser")

        # Detectar articulos restringidos

        if obtener_meta(
            soup, "article:content_tier"
        ) in ("locked", "metered"):

            raise ValueError("PAYWALL")

        for script in soup.select(
            'script[type="application/ld+json"]'
        ):

            if re.search(
                r'"isAccessibleForFree"\s*:\s*(?:false|"false")',
                script.get_text(),
                re.I
            ):

                raise ValueError("PAYWALL")

        # TITULO

        title = obtener_meta(soup, "og:title")

        if not title:

            h1 = soup.find("h1")

            if h1:

                title = h1.get_text(strip=True)

        if not title and soup.title:

            title = soup.title.get_text(strip=True)

        # DESCRIPCION

        description = (
            obtener_meta(soup, "og:description")
            or obtener_meta(soup, "description")
        )

        # AUTOR

        author = (
            obtener_meta(soup, "author")
            or obtener_meta(soup, "article:author")
        )

        # FECHA DE PUBLICACION

        published_at = (
            obtener_meta(soup, "article:published_time")
            or obtener_meta(soup, "datePublished")
            or obtener_meta(soup, "date")
        )

        if not published_at:

            time_tag = soup.find("time")

            if time_tag:

                published_at = time_tag.get("datetime")

        # FECHA DE ACTUALIZACION

        updated_at = (
            obtener_meta(soup, "article:modified_time")
            or obtener_meta(soup, "dateModified")
        )

        # IMAGEN

        image_url = obtener_meta(soup, "og:image")

        if image_url:

            image_url = urljoin(final_url, image_url)

        # CATEGORIA

        category = obtener_meta(
            soup,
            "article:section"
        )

        # URL CANONICA

        canonical_tag = soup.select_one(
            'link[rel~="canonical"][href]'
        )

        canonical_url = final_url

        if canonical_tag:

            candidate = urljoin(
                final_url,
                canonical_tag["href"]
            )

            if (
                urlparse(candidate).hostname
                == urlparse(final_url).hostname
            ):

                canonical_url = candidate

        # CONTENIDO PRINCIPAL

        content, selector = extraer_contenido(soup)

        word_count = len(content.split())

        result.update({
            "canonical_url": canonical_url,
            "title": title,
            "description": description,
            "content": content,
            "author": author,
            "published_at": published_at,
            "updated_at": updated_at,
            "image_url": image_url,
            "category": category,
            "word_count": word_count,
            "selector_used": selector
        })

        # VALIDACION

        if not title or word_count < 30:

            raise ValueError("INSUFFICIENT_CONTENT")

        content_hash = hashlib.sha256(
            " ".join(content.split()).encode("utf-8")
        ).hexdigest()

        result.update({
            "ok": True,
            "content_hash": content_hash,
            "scraping_status": "SUCCESS"
        })

    except requests.exceptions.Timeout:

        result["scraping_status"] = "TIMEOUT"
        result["error"] = "Timeout"

    except ValueError as e:

        result["scraping_status"] = str(e)
        result["error"] = str(e)

    except Exception as e:

        result["scraping_status"] = "ERROR"
        result["error"] = str(e)

    result["scraped_at"] = datetime.now(
        timezone.utc
    ).isoformat()

    return result


# =====================================================
# PRUEBA LOCAL
# =====================================================

if __name__ == "__main__":

    url = "https://t.co/AtNC9avnPS"

    result = scrape_article(url)

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=4
        )
    )