if (!("ontouchstart" in document.documentElement)) {
    document.documentElement.className += "no-touch";
}

$('a').on("touchstart", function (e) {
    var link = $(this);
    if (link.hasClass('hover')) {
        return true;
    } else {
        link.addClass('hover');
        $('a').not(this).removeClass('hover');
        e.preventDefault();
        return false;
    }
});

$(document).ready(function($) {
    $('.container--grid ul').mixItUp({
        animation: {
        duration: 350,
        effects: 'fade translateY(40px)',
        easing: 'ease'
        }
    });
    var filter_tab = $('.filter .placeholder a');
    $('.filter ').on('click', function(event){
        var selected_filter = $(event.target).data('type');
        if( filter_tab.data('type') !== selected_filter ) {
            $('.filter .selected').removeClass('selected');
            $(event.target).addClass('selected');
        }
    });
});


var lazyLoad = {
    getoffsetTop: function(elem) {
        var offTop = 0;
        while (elem != null) {
            offTop += elem.offsetTop;
            elem = elem.offsetParent;
        }
        return offTop;
    },
    isLoad: function(elem) {
        var cHeight = window.innerHeight || document.body.clientHeight || document.documentElement.clientHeight,
            sX = window.pageXOffset || document.body.scrollTop || document.documentElement.scrollTop,
            threshold = 0,
            oTop = this.getoffsetTop(elem),
            viewHeight = oTop - sX - threshold;
        if (cHeight >= viewHeight)
        {
            return true;
        } else {
            return false;
        }
    },
    loadImg: function(elems, isFadein) {
        for (var i = 0, len = elems.length; i < len; i++) {
            if (this.isLoad(elems[i])) {
                var classNames = elems[i].className;
                if (classNames.indexOf("loaded") < 0) {
                    var dataUrl = elems[i].getAttribute("data-url");
                    elems[i].style.backgroundImage = "url(" + dataUrl + ")";
                    elems[i].className = classNames + " loaded";
                    if (isFadein) {
                        this.fadeIn(elems[i]);
                    }
                }
            } else {
                return;
            }
        }
    },
    fadeIn: function(elem) {
        var n = 0,
            isnotIE = window.XMLHttpRequest ? true : false;
        if (isnotIE) {
            elem.style.opacity = 0;
        } else {
            elem.style.filter = "alpha(opacity=0)";
        }
        var t = setInterval(function() {
            if (n < 100) {
                n += 5;
                if (isnotIE) {
                    elem.style.opacity = n / 100;
                } else {
                    elem.style.filter = "alpha(opacity=" + n + ")";
                }
            } else {
                clearInterval(t);
            }
        }, 10);
    }
};

window.onscroll = function() {
    lazyLoad.loadImg(document.getElementsByClassName("card-image"), true);
};
window.onload = function() {
    lazyLoad.loadImg(document.getElementsByClassName("card-image"), true);
};

var getUrlParam = function(name) {
    var results = new RegExp('[\?&]' + name + '=([^&#]*)').exec(decodeURI(window.location.href));
    return results && results[1] || 0;
};

SimpleJekyllSearch.init({
  searchInput: $('#search')[0],
  resultsContainer: $('#content')[0],
  json: '/search.json',
  searchResultTemplate: '<div class="result"><a href="{url}">{title}</a><i id=result-time>Posted on {date}</i></div>',
  noResultsText: '<div class="noresult">No results found</div>',
})

// Staticman comment replies
// modified from Wordpress https://core.svn.wordpress.org/trunk/wp-includes/js/comment-reply.js
// Released under the GNU General Public License - https://wordpress.org/about/gpl/
var addComment = {
  moveForm: function(commId, parentId, respondId, postId) {
    var div,
      element,
      style,
      cssHidden,
      t = this,
      comm = t.I(commId),
      respond = t.I(respondId),
      cancel = t.I("cancel-comment-reply-link"),
      parent = t.I("comment-replying-to"),
      post = t.I("comment-post-slug"),
      commentForm = respond.getElementsByTagName("form")[0];

    if (!comm || !respond || !cancel || !parent || !commentForm) {
      return;
    }

    t.respondId = respondId;
    postId = postId || false;

    if (!t.I("sm-temp-form-div")) {
      div = document.createElement("div");
      div.id = "sm-temp-form-div";
      div.style.display = "none";
      respond.parentNode.insertBefore(div, respond);
    }

    comm.parentNode.insertBefore(respond, comm.nextSibling);
    if (post && postId) {
      post.value = postId;
    }
    parent.value = parentId;
    cancel.style.display = "";

    cancel.onclick = function() {
      var t = addComment,
        temp = t.I("sm-temp-form-div"),
        respond = t.I(t.respondId);

      if (!temp || !respond) {
        return;
      }

      t.I("comment-replying-to").value = null;
      temp.parentNode.insertBefore(respond, temp);
      temp.parentNode.removeChild(temp);
      this.style.display = "none";
      this.onclick = null;
      return false;
    };

    /*
     * Set initial focus to the first form focusable element.
     * Try/catch used just to avoid errors in IE 7- which return visibility
     * 'inherit' when the visibility value is inherited from an ancestor.
     */
    try {
      for (var i = 0; i < commentForm.elements.length; i++) {
        element = commentForm.elements[i];
        cssHidden = false;

        // Modern browsers.
        if ("getComputedStyle" in window) {
          style = window.getComputedStyle(element);
          // IE 8.
        } else if (document.documentElement.currentStyle) {
          style = element.currentStyle;
        }

        /*
         * For display none, do the same thing jQuery does. For visibility,
         * check the element computed style since browsers are already doing
         * the job for us. In fact, the visibility computed style is the actual
         * computed value and already takes into account the element ancestors.
         */
        if (
          (element.offsetWidth <= 0 && element.offsetHeight <= 0) ||
          style.visibility === "hidden"
        ) {
          cssHidden = true;
        }

        // Skip form elements that are hidden or disabled.
        if ("hidden" === element.type || element.disabled || cssHidden) {
          continue;
        }

        element.focus();
        // Stop after the first focusable element.
        break;
      }
    } catch (er) {}

    return false;
  },

  I: function(id) {
    return document.getElementById(id);
  }
};

// Static comments
(function($) {
  var $comments = $(".js-comments");

  $("#comment-form").submit(function() {
    var form = this;

    $(form).addClass("disabled");
    $("#comment-form-submit").val('Loading...');

    $.ajax({
      type: $(this).attr("method"),
      url: $(this).attr("action"),
      data: $(this).serialize(),
      contentType: "application/x-www-form-urlencoded",
      success: function(data) {
        $("#comment-form-submit")
          .val("Submitted")
          .addClass("btn--disabled");
        $("#comment-form .js-notice")
          .removeClass("notice--danger")
          .addClass("notice--success");
        showAlert(
          '<strong>Thanks for your comment!</strong> It is currently pending and will show on the site once approved.'
        );
      },
      error: function(err) {
        console.log(err);
        $("#comment-form-submit").val("Submit Comment");
        $("#comment-form .js-notice")
          .removeClass("notice--success")
          .addClass("notice--danger");
        showAlert(
          "<strong>Sorry, there was an error with your submission.</strong> Please make sure all required fields have been completed and try again."
        );
        $(form).removeClass("disabled");
      }
    });

    return false;
  });

  function showAlert(message) {
    $("#comment-form .js-notice").removeClass("hidden");
    $("#comment-form .js-notice-text").html(message);
  }
})(jQuery);

// ===== Return to Top Button ====
$(window).scroll(function() {
    if ($(this).scrollTop() >= 50) {        // If page is scrolled more than 50px
        $('#return-to-top').fadeIn(200);    // Fade in the arrow
    } else {
        $('#return-to-top').fadeOut(200);   // Else fade out the arrow
    }
});
$('#return-to-top').click(function() {      // When arrow is clicked
    $('body,html').animate({
        scrollTop : 0                       // Scroll to top of body
    }, 500);
});
