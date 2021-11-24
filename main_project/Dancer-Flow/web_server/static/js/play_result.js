
function create_polygon_chart(el, data){
  let sides = 5;
  console.log(el)
  console.log(data)
  let chart = new window.PolygonChart({
    target: el,
    radius: 200,
    data: {
      data: data,
      sides: sides,
      tooltips: {
        roundTo: 2,
        percentage: true,
      },
      colors: {
        normal: {
          polygonStroke: "#A54AE9",
          polygonFill: "#a54ae94f",
          pointStroke: "transparent",
          pointFill: "#A54AE9",
        },
        onHover: {
          polygonStroke: "#A54AE9",
          polygonFill: "#4D02688f",
          pointStroke: "#A54AE9",
          pointFill: "#ffffff",
        },
      },
    },
    polygon: {
      colors: {
        normal: {
          fill: "#231E2C",
          stroke: "#3A255E",
        },
        onHover: {
          fill: "#231E2C",
          stroke: "#993DE0",
        }
      }
    },
    levels: {
      count: 5,
      labels: {
        enabled: true,
        position: {
          spline: 1,
          quadrant: 0,
        },
        colors: {
          normal: "#8B27D7",
          onHover: "#A54AE9",
        },
      },
    },
  });
  return chart
}
// create_polygon_chart(polygon_chart_el)