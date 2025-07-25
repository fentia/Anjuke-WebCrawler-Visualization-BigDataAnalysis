// ...以下为原index.html中<script>标签内的全部JS代码...

// 初始化所有图表
document.addEventListener('DOMContentLoaded', function() {
    // 价格区间分布柱状图
    const priceChart = echarts.init(document.getElementById('priceDistribution'));
    priceChart.setOption({
        tooltip: {
            trigger: 'axis',
            axisPointer: {
                type: 'shadow'
            }
        },
        xAxis: {
            type: 'category',
            data: ['30万以下', '30-40万', '40-50万', '50-60万', '60-80万', '80-100万', '100万以上']
        },
        yAxis: {
            type: 'value',
            name: '房源数量'
        },
        series: [{
            name: '房源数量',
            type: 'bar',
            data: [446, 1072, 1686, 2248, 1943, 501, 338],
            itemStyle: {
                color: '#5470C6'
            }
        }]
    });

    // 户型分布饼图
    const layoutChart = echarts.init(document.getElementById('layoutDistribution'));
    layoutChart.setOption({
        tooltip: {
            trigger: 'item',
            formatter: '{a} <br/>{b}: {c} ({d}%)'
        },
        legend: {
            orient: 'vertical',
            right: 10,
            top: 'center',
            data: ['3室2厅2卫', '4室2厅2卫', '3室1厅2卫', '2室2厅1卫', '其他']
        },
        series: [
            {
                name: '户型分布',
                type: 'pie',
                radius: ['40%', '70%'],
                center: ['40%', '50%'],
                data: [
                    {value: 5051, name: '3室2厅2卫'},
                    {value: 1310, name: '4室2厅2卫'},
                    {value: 18, name: '3室1厅2卫'},
                    {value: 410, name: '2室2厅1卫'},
                    {value: 1445, name: '其他'},
                ],
                emphasis: {
                    itemStyle: {
                        shadowBlur: 10,
                        shadowOffsetX: 0,
                        shadowColor: 'rgba(0, 0, 0, 0.5)'
                    }
                },
                label: {
                    show: false,
                    position: 'center'
                },
                labelLine: {
                    show: false
                }
            }
        ]
    });

    // 区域均价对比雷达图
    const districtChart = echarts.init(document.getElementById('districtPrice'));
    districtChart.setOption({
        tooltip: {},
        legend: {
            data: ['均价(元/㎡)'],
            bottom: 10
        },
        radar: {
            indicator: [
                { name: '高坪区', max: 8000 },
                { name: '顺庆区', max: 8000 },
                { name: '嘉陵区', max: 8000 },
                { name: '阆中市', max: 8000 },
                { name: '营山县', max: 8000 },
                { name: '南部县', max: 8000 },
                { name: '蓬安县', max: 8000 },
                { name: '仪陇县', max: 8000 },
                { name: '西充县', max: 8000 }
            ],
            splitArea: {
                show: false
            }
        },
        series: [{
            name: '区域均价对比',
            type: 'radar',
            data: [
                {
                    value: ['5622.25', '6842.08', '5035.69', '4617.09', '4619.83', '5114.64', '4947.58', '4153.88', '5127.61'],
                    name: '均价(元/㎡)',
                    areaStyle: {
                        color: 'rgba(121, 134, 203, 0.4)'
                    },
                    lineStyle: {
                        width: 2
                    }
                }
            ]
        }]
    });

    // 建造年份分布折线图
    const yearChart = echarts.init(document.getElementById('yearDistribution'));
    yearChart.setOption({
        tooltip: {
            trigger: 'axis'
        },
        xAxis: {
            type: 'category',
            data: ['1992年', '1993年', '1995年', '1996年', '1997年', '1998年', '1999年', '2000年', '2001年', '2002年', '2003年', '2004年', '2005年', '2006年', '2007年', '2008年', '2009年', '2010年', '2011年', '2012年', '2013年', '2014年', '2015年', '2016年', '2017年', '2018年', '2019年', '2020年', '2021年', '2022年', '2023年', '2024年', '2025年']
        },
        yAxis: {
            type: 'value',
            name: '房源数量'
        },
        series: [{
            name: '房源数量',
            type: 'line',
            data: [2, 1, 11, 12, 1, 17, 10, 125, 17, 20, 27, 23, 60, 68, 67, 129, 86, 231, 66, 358, 422, 365, 791, 472, 320, 504, 487, 996, 646, 814, 703, 317, 66],
            smooth: true,
            lineStyle: {
                width: 3,
                color: '#91CC75'
            },
            itemStyle: {
                color: '#91CC75'
            },
            areaStyle: {
                color: {
                    type: 'linear',
                    x: 0,
                    y: 0,
                    x2: 0,
                    y2: 1,
                    colorStops: [{
                        offset: 0,
                        color: 'rgba(145, 204, 117, 0.5)'
                    }, {
                        offset: 1,
                        color: 'rgba(145, 204, 117, 0.1)'
                    }]
                }
            }
        }]
    });

    // 建造年份与均价关系折线图
    const priceYearChart = echarts.init(document.getElementById('priceYearRelation'));
    priceYearChart.setOption({
        tooltip: {
            trigger: 'axis'
        },
        xAxis: {
            type: 'category',
            data: ['1992年', '1993年', '1995年', '1996年', '1997年', '1998年', '1999年', '2000年', '2001年', '2002年', '2003年', '2004年', '2005年', '2006年', '2007年', '2008年', '2009年', '2010年', '2011年', '2012年', '2013年', '2014年', '2015年', '2016年', '2017年', '2018年', '2019年', '2020年', '2021年', '2022年', '2023年', '2024年', '2025年']
        },
        yAxis: {
            type: 'value',
            name: '均价(元/㎡)',
        },
        series: [{
            name: '均价(元/㎡)',
            type: 'line',
            data: ['3311.50', '5767.00', '4733.55', '5851.00', '4237.00', '3258.12', '3692.10', '4159.02', '3554.12', '3460.35', '3717.07', '3273.13', '3324.67', '3742.91', '3553.13', '3834.99', '4340.17', '4749.09', '4919.82', '5255.25', '5255.61', '5450.33', '5331.75', '5466.88', '5561.40', '5732.96', '5670.22', '5693.17', '5595.42', '5671.56', '9827.17', '5712.42', '5355.95'],
            smooth: true,
            lineStyle: {
                width: 3,
                color: '#91CC75'
            },
            itemStyle: {
                color: '#91CC75'
            },
            areaStyle: {
                color: {
                    type: 'linear',
                    x: 0,
                    y: 0,
                    x2: 0,
                    y2: 1,
                    colorStops: [{
                        offset: 0,
                        color: 'rgba(145, 204, 117, 0.5)'
                    }, {
                        offset: 1,
                        color: 'rgba(145, 204, 117, 0.1)'
                    }]
                }
            }
        }]
    });

    // 房产均价与面积关系散点图
    const priceAreaChart = echarts.init(document.getElementById('priceAreaRelation'));
    const priceAreaOption = {
        tooltip: {
            formatter: function (param) {
                return param.data[2] + '<br>面积: ' + param.data[0] + '㎡<br>均价: ' + param.data[1] + '元/㎡';
            }
        },
        grid: {
            left: '3%',
            right: '7%',
            bottom: '7%',
            containLabel: true
        },
        xAxis: {
            name: '面积(㎡)',
            type: 'value',
            max: 450
        },
        yAxis: {
            name: '均价(元/㎡)',
            type: 'value',
            min: 0,
            max: 1000000
        },
        visualMap: {
            min: 3000,
            max: 8000,
            dimension: 1,
            orient: 'horizontal',
            right: 'center',
            top: '1%',
            inRange: {
                color: ['#50a3ba', '#eac736', '#d94e5d']
            },
            text: ['高均价', '低均价'],
            calculable: true
        },
        dataZoom: [
            {
                type: 'slider',
                yAxisIndex: 0,
                filterMode: 'none',
                startValue: 0,
                endValue: 20000,
                width: 16,
                right: 10
            }
        ],
        series: [{
            name: '均价-面积',
            type: 'scatter',
            symbolSize: function(data) {
                return Math.sqrt(data[0]) * 1.5;
            },
            data: [],
            itemStyle: {
                opacity: 0.8,
                borderColor: '#fff',
                borderWidth: 1
            }
        }]
    };
    priceAreaChart.setOption(priceAreaOption);

    // 使用$.get加载数据
    $.get('data//avgPriceAndArea.json', function (data) {
        // data为数组，每项为[面积, 均价, 小区名]
        priceAreaOption.series[0].data = data;
        priceAreaChart.setOption(priceAreaOption);
    });

    // 各小区房源数量TOP10
    const communityChart = echarts.init(document.getElementById('communityTop10'));
    communityChart.setOption({
        tooltip: {
            trigger: 'axis',
            axisPointer: {
                type: 'shadow'
            }
        },
        xAxis: {
            type: 'value',
            name: '房源数量'
        },
        yAxis: {
            type: 'category',
            data: ['蓝光香江国际', '阳光江山公园城', '碧桂园原树缇香', '铁投锦华府', '明宇帝壹家西区', '春风玫瑰园', '江东首席', '碧桂园天樾', '华邦天誉', '泰合尚渡'],
            axisLabel: {
                interval: 0,
                rotate: 0
            }
        },
        series: [{
            name: '房源数量',
            type: 'bar',
            data: [129, 93, 74, 69, 69, 62, 61, 60, 59, 59],
            itemStyle: {
                color: function(params) {
                    var colorList = ['#c23531','#2f4554','#61a0a8','#d48265','#91c7ae','#749f83','#ca8622','#bda29a','#6e7074','#546570'];
                    return colorList[params.dataIndex];
                }
            },
            label: {
                show: true,
                position: 'right'
            }
        }]
    });

    
    // 窗口大小变化时重新调整图表大小
    window.addEventListener('resize', function() {
        priceChart.resize();
        layoutChart.resize();
        districtChart.resize();
        yearChart.resize();
        priceAreaChart.resize();
        communityChart.resize();
    });
});

// 图片点击放大功能
document.addEventListener('DOMContentLoaded', function() {
    // 图片点击放大功能
    document.querySelectorAll('.enlarge-img').forEach(function(img) {
        img.addEventListener('click', function() {
            var modal = document.getElementById('imgModal');
            var modalImg = document.getElementById('imgModalImg');
            modal.style.display = 'flex';
            modalImg.src = img.getAttribute('data-src') || img.src;
        });
    });
    document.getElementById('imgModalClose').onclick = function(e) {
        document.getElementById('imgModal').style.display = 'none';
        e.stopPropagation(); // 防止冒泡到模态框
    };
    // 点击模态框外部关闭
    document.getElementById('imgModal').onclick = function(e) {
        if (e.target === this) this.style.display = 'none';
    };
});